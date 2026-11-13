"""Last-block CLS value-path sensitivity on a fixed Oxford Pets subset."""

from __future__ import annotations

import hashlib
import json
import argparse
import sys
import time
from contextlib import contextmanager
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs/rerun/cls"
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

from fastweight_vit.execution.execute import split_qkv  # noqa: E402
from fastweight_vit.models.deit import CHECKPOINTS, load_deit  # noqa: E402
from fastweight_vit.readings import capture  # noqa: E402
from experiments.spatial.run_spatial import PetTest  # noqa: E402


N_IMAGES = 128
BATCH_SIZE = 4
N_PATCHES = 196
VALIDATION_PATCHES = (0, 97, 195)
SEED = 2026
METHODS = ("all_r0", "all_c3", "cls_attention", "cls_sum_head", "cls_net")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_names(output: Path) -> list[str]:
    source = ROOT / "experiments/spatial/selected_names.txt"
    names = [name.strip() for name in source.read_text().splitlines() if name.strip()]
    if len(names) != 384 or len(set(names)) != 384:
        raise ValueError("Expected the 384 fixed Oxford spatial-study image names")
    names.sort(key=lambda name: hashlib.sha256(f"v13-cls-2026:{name}".encode()).digest())
    chosen = names[:N_IMAGES]
    if chosen != (HERE / "selected_names.txt").read_text(encoding="utf-8").splitlines():
        raise ValueError("CLS selection differs from the archived fixed cohort")
    (output / "selected_names.txt").write_text("\n".join(chosen) + "\n", encoding="utf-8")
    return chosen


@contextmanager
def capture_input(block):
    observed = {}

    def hook(_module, inputs):
        observed["x"] = inputs[0]

    handle = block.register_forward_pre_hook(hook)
    try:
        yield observed
    finally:
        handle.remove()


def cls_after_attention(model, block, initial_cls, attended_cls):
    """Run the exact remaining token-wise block path and the model head."""
    x = initial_cls + block.drop_path1(block.ls1(attended_cls))
    x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))
    flat = x.reshape(-1, 1, x.shape[-1])
    logits = model.forward_head(model.norm(flat))
    return logits.reshape(*x.shape[:-1], -1)


def score_batch(model, block, images):
    with capture_input(block) as observed:
        original = model(images)
    x = observed["x"]
    attn = block.attn
    if not isinstance(attn.norm, torch.nn.Identity):
        raise ValueError("Expected standard DeiT-S attention output norm")
    if len(x.shape) != 3 or x.shape[1] != N_PATCHES + 1:
        raise ValueError("Expected CLS plus 196 patch tokens")

    q, k, v = split_qkv(attn, block.norm1(x))
    p = ((q * attn.scale) @ k.transpose(-2, -1)).softmax(-1)
    p_cls = p[:, :, 0, :]
    weighted = torch.einsum("bhn,bhne->bhe", p_cls, v).reshape(len(images), -1)
    # Each row has the same original CLS attention distribution; subtract only
    # a single source value term before the original output projection/bias.
    source_terms = (p_cls[:, :, 1:, None] * v[:, :, 1:, :]).permute(0, 2, 1, 3)
    source_terms = source_terms.reshape(len(images), N_PATCHES, -1)
    modified = attn.proj_drop(attn.proj(attn.norm(weighted[:, None, :] - source_terms)))
    deleted_logits = cls_after_attention(model, block, x[:, 0, None, :], modified)
    baseline_attention = attn.proj_drop(attn.proj(attn.norm(weighted)))
    reconstructed = cls_after_attention(model, block, x[:, 0, :], baseline_attention)
    baseline_error = float((reconstructed - original).abs().max())
    if baseline_error > 2e-3:
        raise AssertionError(f"Original logits not reconstructed: {baseline_error:.6g}")

    head_weight = attn.proj.weight.reshape(-1, attn.num_heads, attn.head_dim)
    projected_values = torch.einsum("dhe,bhje->bhjd", head_weight, v[:, :, 1:, :])
    head_magnitude = projected_values.norm(dim=-1)
    received = p[:, :, :, 1:].sum(dim=2)
    cls_weights = p_cls[:, :, 1:]
    scores = {
        "all_r0": received.sum(dim=1),
        "all_c3": (received * head_magnitude).sum(dim=1),
        "cls_attention": cls_weights.sum(dim=1),
        "cls_sum_head": (cls_weights * head_magnitude).sum(dim=1),
        "cls_net": (cls_weights[:, :, :, None] * projected_values).sum(dim=1).norm(dim=-1),
    }
    classes = original.topk(2, dim=-1).indices
    top, runner = classes[:, 0], classes[:, 1]
    original_margin = original.gather(1, top[:, None])[:, 0] - original.gather(1, runner[:, None])[:, 0]
    changed_margin = (deleted_logits.gather(2, top[:, None, None].expand(-1, N_PATCHES, 1))[:, :, 0]
                      - deleted_logits.gather(2, runner[:, None, None].expand(-1, N_PATCHES, 1))[:, :, 0])
    effect = (changed_margin - original_margin[:, None]).abs()
    if not torch.isfinite(effect).all() or any(not torch.isfinite(value).all() for value in scores.values()):
        raise ValueError("Non-finite CLS effect or ranking score")
    return {"original": original, "deleted": deleted_logits, "effect": effect,
            "scores": scores, "top": top, "runner": runner, "original_margin": original_margin,
            "baseline_error": baseline_error}


@contextmanager
def hook_cls_value(attn, source_patch):
    """Independent full-model forward with one final-block CLS source value zeroed."""
    original = attn.__dict__.get("forward")

    def replacement(x, attn_mask=None, is_causal=False):
        if attn_mask is not None or is_causal:
            raise ValueError("Expected unmasked noncausal DeiT attention")
        batch, tokens, _ = x.shape
        q, k, v = split_qkv(attn, x)
        p = ((q * attn.scale) @ k.transpose(-2, -1)).softmax(-1)
        head_out = p @ v
        head_out = head_out.clone()
        head_out[:, :, 0, :] -= p[:, :, 0, source_patch + 1, None] * v[:, :, source_patch + 1, :]
        full = head_out.transpose(1, 2).reshape(batch, tokens, -1)
        return attn.proj_drop(attn.proj(attn.norm(full)))

    attn.forward = replacement
    try:
        yield
    finally:
        if original is None:
            del attn.forward
        else:
            attn.forward = original


def spearman(a, b):
    ra, rb = rankdata(a, method="average"), rankdata(b, method="average")
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = np.linalg.norm(ra) * np.linalg.norm(rb)
    if denom == 0:
        raise ValueError("A per-image 196-source ranking is constant")
    return float((ra @ rb) / denom)


def summarize(per_image: pd.DataFrame):
    rng = np.random.default_rng(SEED)
    draws = rng.integers(len(per_image), size=(10000, len(per_image)))
    rows = []
    for index, method in enumerate(METHODS):
        values = per_image[method].to_numpy()
        means = values[draws].mean(axis=1)
        rows.append({"kind": "method", "a": method, "b": "", "n": len(values),
                     "mean": float(values.mean()), "low": float(np.quantile(means, .025)),
                     "high": float(np.quantile(means, .975))})
        for other in METHODS[:index]:
            diff = values - per_image[other].to_numpy()
            means = diff[draws].mean(axis=1)
            rows.append({"kind": "paired_difference", "a": method, "b": other, "n": len(diff),
                         "mean": float(diff.mean()), "low": float(np.quantile(means, .025)),
                         "high": float(np.quantile(means, .975))})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT,
                        help="New run output directory (default: outputs/rerun/cls)")
    parser.add_argument("--model", choices=("small", "base"), default="small")
    parser.add_argument("--protocol", type=Path, default=HERE / "protocol.md",
                        help="Protocol fixed before this run; saved by hash in metadata")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output == HERE:
        parser.error("--output-dir must not overwrite the archived exploratory results")
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for the full 128-image study")
    names = selected_names(output)
    model = load_deit("cuda", base=args.model == "base")
    dataset = PetTest(model)
    spatial_names = (ROOT / "experiments/spatial/selected_names.txt").read_text().splitlines()
    if dataset.names != spatial_names:
        raise ValueError("Current Oxford files do not reproduce the fixed spatial-study selection")
    if not set(names) <= set(dataset.names):
        raise ValueError("CLS names are not in the fixed spatial selection")
    indices = [dataset.names.index(name) for name in names]
    block = model.blocks[11]
    source_rows, image_rows, validation_rows = [], [], []
    reference_error = 0.0
    max_baseline_error = 0.0
    for start_index in range(0, N_IMAGES, BATCH_SIZE):
        part_names = names[start_index:start_index + BATCH_SIZE]
        part_indices = indices[start_index:start_index + BATCH_SIZE]
        images = torch.stack([dataset[index][0] for index in part_indices]).cuda()
        batch = score_batch(model, block, images)
        max_baseline_error = max(max_baseline_error, batch["baseline_error"])
        if start_index == 0:
            with capture(model, (12,), per_head=True) as reference:
                check_logits = model(images)
            reference_error = max(float((check_logits - batch["original"]).abs().max()),
                                  float((reference[12]["r0"] - batch["scores"]["all_r0"]).abs().max()),
                                  float((reference[12]["c3"] - batch["scores"]["all_c3"]).abs().max()))
            if reference_error > 2e-3:
                raise AssertionError(f"Reference R0/C3 mismatch: {reference_error:.6g}")
            for patch in VALIDATION_PATCHES:
                with hook_cls_value(block.attn, patch):
                    checked_batch = model(images)
                for local_index in range(4):
                    checked = checked_batch[local_index]
                    predicted = batch["deleted"][local_index, patch]
                    max_logit_error = float((checked - predicted).abs().max())
                    top = batch["top"][local_index]
                    runner = batch["runner"][local_index]
                    margin_error = float(abs((checked[top] - checked[runner])
                                             - (predicted[top] - predicted[runner])))
                    if max_logit_error > 2e-3 or margin_error > 2e-3:
                        raise AssertionError(f"Full forward mismatch for {part_names[local_index]}, j={patch}: "
                                             f"logit={max_logit_error:.6g}, margin={margin_error:.6g}")
                    validation_rows.append({"image": part_names[local_index], "source_patch": patch,
                                            "max_logit_error": max_logit_error,
                                            "fixed_margin_error": margin_error})
        effects = batch["effect"].cpu().numpy()
        scores = {name: value.cpu().numpy() for name, value in batch["scores"].items()}
        for local_index, name in enumerate(part_names):
            image_row = {"image": name, "spatial_index": part_indices[local_index],
                         "original_top1": int(batch["top"][local_index]),
                         "original_runnerup": int(batch["runner"][local_index]),
                         "original_margin": float(batch["original_margin"][local_index]),
                         "mean_abs_margin_change": float(effects[local_index].mean())}
            for method in METHODS:
                image_row[method] = spearman(effects[local_index], scores[method][local_index])
            image_rows.append(image_row)
            for patch in range(N_PATCHES):
                source_rows.append({"image": name, "source_patch": patch,
                                    "abs_margin_change": float(effects[local_index, patch]),
                                    **{method: float(scores[method][local_index, patch]) for method in METHODS}})
        if start_index % 16 == 0:
            print(f"Processed {start_index + len(part_names)}/{N_IMAGES}", flush=True)

    per_image = pd.DataFrame(image_rows)
    per_source = pd.DataFrame(source_rows)
    if len(per_image) != N_IMAGES or len(per_source) != N_IMAGES * N_PATCHES:
        raise ValueError("Incomplete output grid")
    per_image.to_csv(output / "per_image.csv", index=False)
    per_source.to_csv(output / "per_source.csv.gz", index=False,
                      compression={"method": "gzip", "mtime": 0})
    summary = summarize(per_image)
    summary.to_csv(output / "summary.csv", index=False)
    pd.DataFrame(validation_rows).to_csv(output / "forward_validation.csv", index=False)
    meta = {"n_images": N_IMAGES, "n_source_patches": N_PATCHES, "batch_size": BATCH_SIZE,
            "selection_seed_string": "v13-cls-2026:",
            "spatial_selection_sha256": sha256(ROOT / "experiments/spatial/selected_names.txt"),
            "cls_selection_sha256": sha256(output / "selected_names.txt"),
            "protocol_sha256": sha256(args.protocol),
            "script_sha256": sha256(Path(__file__)),
            "spatial_loader_sha256": sha256(ROOT / "experiments/spatial/run_spatial.py"),
            "model_source_sha256": sha256(ROOT / "src/fastweight_vit/models/deit.py"),
            "checkpoint": CHECKPOINTS[args.model], "model_eval": not model.training,
            "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "max_baseline_logit_error": max_baseline_error,
            "max_reference_logit_or_map_error": reference_error,
            "full_forward_validations": len(validation_rows),
            "max_full_forward_logit_error": max(row["max_logit_error"] for row in validation_rows),
            "max_full_forward_margin_error": max(row["fixed_margin_error"] for row in validation_rows),
            "elapsed_seconds": time.perf_counter() - start}
    (output / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
