"""Exploratory spatial controls for frozen supervised DeiT-S; see PROTOCOL.md.

Run from the repository root with the project's vb Python and PYTHONPATH
containing src and the existing build/paper-v9/python-deps overlay. Oxford
images and annotations must be extracted in build/paper-v13/spatial.
"""

import csv
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from fastweight_vit.data.masks import MaskDataset
from fastweight_vit.enrichment import measures
from fastweight_vit.execution.execute import split_qkv
from fastweight_vit.models.deit import CHECKPOINTS, load_deit
from fastweight_vit.readings import capture
from fastweight_vit.tokens import centre_rings


ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "build/paper-v13/spatial"
RESULTS = Path(__file__).resolve().parent
LAYERS = (9, 10, 11, 12)
METHODS = ("R0", "C3", "C3v", "norm_only", "donor_rank", "ring_shuffle", "net_output")
N_IMAGES = 384
SEED = 2026


class PetTest(MaskDataset):
    def __init__(self, model):
        super().__init__(model)
        entries = (CACHE / "annotations/test.txt").read_text().splitlines()
        names = [line.split()[0] for line in entries if line.strip()]
        names.sort(key=lambda name: hashlib.sha256(f"v13-spatial-2026:{name}".encode()).digest())
        self.names = names[:N_IMAGES]
        if len(self.names) != N_IMAGES or len(set(self.names)) != N_IMAGES:
            raise ValueError("Missing or duplicate Oxford test names")
        missing = [name for name in self.names if not (CACHE / "images" / f"{name}.jpg").is_file()
                   or not (CACHE / "annotations/trimaps" / f"{name}.png").is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} selected image/trimap pairs, e.g. {missing[:3]}")

    def __len__(self):
        return len(self.names)

    def sample(self, index):
        name = self.names[index]
        with Image.open(CACHE / "images" / f"{name}.jpg") as image:
            rgb = image.convert("RGB")
        with Image.open(CACHE / "annotations/trimaps" / f"{name}.png") as trimap:
            mask = trimap.convert("L")
        return rgb, -1, mask


def ring_permutation(layer, index, device):
    """One within-ring permutation shared by heads, deterministic by image/layer."""
    rings = centre_rings()[1:].numpy()
    order = np.arange(196)
    rng = np.random.default_rng(SEED * 1_000_003 + 1009 * index + layer)
    for ring in np.unique(rings):
        members = np.flatnonzero(rings == ring)
        order[members] = rng.permutation(members)
    return torch.from_numpy(order).to(device)


@contextmanager
def capture_spatial(model, image_indices):
    """Collect per-head readings while preserving the original model forward."""
    saved = {}
    store = {}

    def replacement(layer, attention):
        weight = attention.proj.weight.reshape(-1, attention.num_heads, attention.head_dim)

        def forward(x, attn_mask=None, is_causal=False):
            if attn_mask is not None or is_causal or not isinstance(attention.norm, torch.nn.Identity):
                raise ValueError("Expected standard unmasked DeiT self-attention")
            batch, tokens, _ = x.shape
            q, k, v = split_qkv(attention, x)
            p = ((q * attention.scale) @ k.transpose(-2, -1)).softmax(-1)
            a = p[:, :, :, 1:]
            z = torch.einsum("dhe,bhje->bhjd", weight, v[:, :, 1:, :])
            length = z.norm(dim=-1)
            received = a.sum(dim=2)
            value_length = v[:, :, 1:, :].norm(dim=-1)
            net = torch.zeros((batch, 196), device=x.device, dtype=x.dtype)
            max_violation = 0.0
            for first in range(0, tokens, 32):
                weights = a[:, :, first:first + 32, :]
                vectors = torch.einsum("bhij,bhjd->bijd", weights, z)
                magnitude = vectors.norm(dim=-1)
                upper = (weights * length[:, :, None, :]).sum(dim=1)
                max_violation = max(max_violation, float((magnitude - upper).max()))
                net += magnitude.sum(dim=1)
            store[layer] = dict(received=received, length=length,
                                value_length=value_length, net=net,
                                max_violation=max_violation)
            out = (p @ v).transpose(1, 2).reshape(batch, tokens, -1)
            return attention.proj_drop(attention.proj(attention.norm(out)))

        return forward

    try:
        for layer in LAYERS:
            attention = model.blocks[layer - 1].attn
            saved[layer] = attention.__dict__.get("forward")
            attention.forward = replacement(layer, attention)
        yield store
    finally:
        for layer, original in saved.items():
            attention = model.blocks[layer - 1].attn
            if original is None:
                del attention.forward
            else:
                attention.forward = original


def scores_for_batch(store, indices):
    batch = len(indices)
    result = {key: torch.zeros((batch, 196), device=store[LAYERS[0]]["received"].device)
              for key in METHODS}
    for layer in LAYERS:
        piece = store[layer]
        a, length, value_length = piece["received"], piece["length"], piece["value_length"]
        if batch != 2:
            raise ValueError("Donor design requires batches of exactly two")
        donor = length.flip(0)
        donor_order = donor.argsort(dim=-1)
        borrowed = torch.empty_like(length).scatter(-1, donor_order, length.sort(dim=-1).values)
        shuffled = torch.stack([length[row, :, ring_permutation(layer, index, length.device)]
                                for row, index in enumerate(indices)])
        result["R0"] += a.sum(dim=1)
        result["C3"] += (a * length).sum(dim=1)
        result["C3v"] += (a * value_length).sum(dim=1)
        result["norm_only"] += (a.mean(dim=-1, keepdim=True) * length).sum(dim=1)
        result["donor_rank"] += (a * borrowed).sum(dim=1)
        result["ring_shuffle"] += (a * shuffled).sum(dim=1)
        result["net_output"] += piece["net"]
    return result


def verify_reference(model, images, own):
    with capture(model, LAYERS, per_head=True) as reference:
        logits = model(images)
    for layer in LAYERS:
        expected = reference[layer]["c3"]
        got = (own[layer]["received"] * own[layer]["length"]).sum(dim=1)
        if not torch.allclose(got, expected, rtol=3e-5, atol=1e-4):
            raise AssertionError(f"C3 mismatch at layer {layer}: {(got - expected).abs().max()}")
    return logits


def summarize(rows):
    names = METHODS
    by_method = {method: {row["index"]: row for row in rows if row["method"] == method}
                 for method in names}
    indices = sorted(set.intersection(*(set(by_method[method]) for method in names)))
    generator = np.random.default_rng(SEED)
    boot_indices = generator.integers(0, len(indices), (10000, len(indices)))
    metrics = ("enrichment", "auroc", "precision", "mass")
    output = {"n": len(indices), "methods": {}}
    for method in names:
        output["methods"][method] = {}
        for metric in metrics:
            values = np.array([float(by_method[method][i][metric]) for i in indices])
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite {method} {metric}")
            entry = {"mean": float(values.mean())}
            for reference in ("R0", "C3"):
                baseline = np.array([float(by_method[reference][i][metric]) for i in indices])
                diff = values - baseline
                intervals = np.quantile(diff[boot_indices].mean(axis=1), [0.025, 0.975])
                entry[f"delta_{reference}"] = float(diff.mean())
                entry[f"delta_{reference}_ci95"] = intervals.tolist()
            output["methods"][method][metric] = entry
    output["object_area_mean"] = float(np.mean([float(by_method["R0"][i]["object_area"])
                                                for i in indices]))
    return output


def main():
    torch.set_grad_enabled(False)
    if not torch.cuda.is_available():
        raise RuntimeError("RTX GPU/CUDA is required for this exploratory run")
    model = load_deit("cuda")
    dataset = PetTest(model)
    (RESULTS / "selected_names.txt").write_text("\n".join(dataset.names) + "\n")
    rows, excluded = [], []
    max_triangle_violation = 0.0
    first_verified = False
    for start in range(0, N_IMAGES, 2):
        indices = [start, start + 1]
        samples = [dataset[index] for index in indices]
        images = torch.stack([sample[0] for sample in samples]).cuda()
        foreground = torch.stack([sample[1] for sample in samples])
        valid = torch.stack([sample[2] for sample in samples])
        with capture_spatial(model, indices) as store:
            logits = model(images)
        if not first_verified:
            reference_logits = verify_reference(model, images, store)
            if not torch.allclose(logits, reference_logits, rtol=3e-5, atol=1e-4):
                raise AssertionError(f"Model output mismatch: {(logits-reference_logits).abs().max()}")
            first_verified = True
        max_triangle_violation = max(max_triangle_violation,
                                     *(store[layer]["max_violation"] for layer in LAYERS))
        if max_triangle_violation > 1e-4:
            raise AssertionError(f"Triangle inequality violation: {max_triangle_violation}")
        scores = scores_for_batch(store, indices)
        for method, tensor in scores.items():
            results = measures(tensor, foreground, valid, foreground.flip(0),
                               generator=torch.Generator().manual_seed(SEED + start))
            for row, index in enumerate(indices):
                if not (foreground[row] & valid[row]).any() or not (valid[row] & ~foreground[row]).any():
                    excluded.append(dataset.names[index])
                    continue
                rows.append({"index": index, "image": dataset.names[index], "method": method,
                             **{key: float(value[row]) for key, value in results.items()}})
        if start % 32 == 0:
            print(f"Processed {start+2}/{N_IMAGES}", flush=True)
    with (RESULTS / "per_image.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    summary.update({"model": CHECKPOINTS["small"], "dataset": "Oxford-IIIT Pet test trimaps",
                    "selected": N_IMAGES, "excluded": sorted(set(excluded)), "layers": LAYERS,
                    "max_triangle_violation": max_triangle_violation,
                    "torch": torch.__version__, "device": torch.cuda.get_device_name(0),
                    "preprocessing": model.pretrained_cfg})
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"n": summary["n"], "enrichment": {
        key: entry["enrichment"] for key, entry in summary["methods"].items()}}, indent=2))


if __name__ == "__main__":
    main()
