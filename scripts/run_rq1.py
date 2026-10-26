"""Run the complete RQ1 sweep with fixed original-model summaries."""

import argparse
import csv
import hashlib
import logging
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from mita_probe.runtime import configure_runtime, data_root, project_path

ROOT = configure_runtime()

import pandas as pd
import timm
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from mita_probe.assets import MODEL_REVISION
from mita_probe.data import ImageNetValidation, Imagenette1k, experiment_indices
from mita_probe.io import load_config, source_commit, write_csv, write_json
from mita_probe.metrics import agreement, kl_divergence
from mita_probe.models import execute_attention, load_model, read_attention
from mita_probe.rq1 import aggregate_predictions, mean_interval, settings, validate_completed_settings


def validate_config(config):
    required = {
        "model": "deit_small_patch16_224.fb_in1k", "rq": "rq1",
        "replace_scope": "all", "mode": "strict", "m": 16,
        "kmeans_iterations": 10, "kmeans_init": "grid", "subset_seed": 0,
    }
    for key, value in required.items():
        if config[key] != value:
            raise ValueError(f"This RQ1 implementation requires {key}={value}")
    if config["dataset"] not in {"imagenette", "imagenet_val"}:
        raise ValueError("Use Imagenette or the saved ImageNet validation split")
    authorized_extra = config["k"] == [128] and config.get("purpose") == "user_requested_k128"
    identity_only = config["k"] == [] and config.get("purpose") == "identity_200"
    if config["k"] != [4, 8, 16, 32, 64] and not authorized_extra and not identity_only:
        raise ValueError("Keep the prespecified k sweep")
    if config["methods"] != ["grid", "random", "kmeans"] or config["random_seeds"] != list(range(5)):
        raise ValueError("Run all three methods and random seeds 0-4")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/rq1_pilot.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    validate_config(config)
    commit = source_commit(ROOT)
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(output / "log.txt", encoding="utf-8"), logging.StreamHandler()],
    )
    try:
        run(config, output, commit, args.resume_from)
    except Exception:
        logging.exception("RQ1 run failed")
        raise


def run(config, output, commit, resume_from=None):
    import json

    torch.set_num_threads(2)
    torch.manual_seed(config["subset_seed"])
    torch.use_deterministic_algorithms(True)
    device = config["device"]
    model = load_model(device)
    dataset = (Imagenette1k(data_root(), model) if config["dataset"] == "imagenette"
               else ImageNetValidation(data_root() / "imagenet_val", model))
    indices = experiment_indices(
        len(dataset), config["n_images"], config["subset_seed"],
        project_path(config["indices"]) if config["indices"] else None,
    )
    write_json(output / "indices.json", indices)
    loader = DataLoader(Subset(dataset, indices), batch_size=config["batch_size"],
                        shuffle=False, num_workers=config["num_workers"])
    timestamp = datetime.now(ZoneInfo("Asia/Singapore")).isoformat()
    metadata = {
        "timestamp": timestamp, "git_commit": commit, "torch": torch.__version__,
        "timm": timm.__version__, "device": torch.cuda.get_device_name() if device == "cuda" else device,
        "preprocessing": dataset.preprocessing, "dtype": "float32, no autocast",
        "explanation_source": "original model, frozen for execution in all 12 layers",
        "random_rng": "independent generator per (k, seed), continuous across batches",
        "indices_sha256": hashlib.sha256((output / "indices.json").read_bytes()).hexdigest(),
        "checkpoint_revision": MODEL_REVISION,
        "purpose": config.get("purpose", "rq1"),
    }
    if config["dataset"] == "imagenet_val":
        metadata["dataset_manifest"] = json.loads((data_root() / "imagenet_manifest.json").read_text())
    if resume_from is not None:
        resume_from = ROOT / resume_from
        if load_config(resume_from / 'config.yaml') != config or config.get('collect_grouping_counts'):
            raise ValueError('Continuation requires identical configuration and no count-histogram collection')
        previous = json.loads((resume_from / 'metadata.json').read_text(encoding='utf-8'))
        for field in ['torch', 'timm', 'device', 'preprocessing', 'dtype', 'indices_sha256', 'checkpoint_revision']:
            # JSON converts tuples in preprocessing to lists.
            if json.loads(json.dumps(metadata[field])) != previous[field]:
                raise ValueError(f'Continuation mismatch: {field}')
        metadata['continued_from'] = str(resume_from.relative_to(ROOT))
        metadata['previous_git_commit'] = previous['git_commit']
    write_json(output / "metadata.json", metadata)
    started = time.perf_counter()
    original_batches, identity_rows = [], []
    max_read_error = max_identity_error = 0.
    with torch.inference_mode():
        offset = 0
        for images, labels in loader:
            images = images.to(device)
            original = model(images)
            with read_attention(model, k_top=197) as patterns:
                observed = model(images)
            with execute_attention(model, patterns):
                identity = model(images)
            max_read_error = max(max_read_error, (observed - original).abs().max().item())
            max_identity_error = max(max_identity_error, (identity - original).abs().max().item())
            if not all(torch.isfinite(x).all() for x in [original, observed, identity]):
                raise FloatingPointError("Non-finite output in identity check")
            original_batches.append(original.cpu())
            for index, label, pred, ident in zip(
                indices[offset:offset + len(images)], labels.tolist(),
                original.argmax(-1).tolist(), identity.argmax(-1).tolist(),
            ):
                identity_rows.append({"index": index, "target": label, "original": pred,
                                      "identity": ident, "agree": int(pred == ident)})
            offset += len(images)
    write_csv(output / "identity_predictions.csv", identity_rows)
    checks = {
        "n_images": len(indices), "identity_agreement": sum(r["agree"] for r in identity_rows) / len(indices),
        "max_read_logit_error": max_read_error, "max_identity_logit_error": max_identity_error,
        "acc_original": sum(r["original"] == r["target"] for r in identity_rows) / len(indices),
        "identity_seconds": time.perf_counter() - started,
    }
    write_json(output / "checks.json", checks)
    if checks["identity_agreement"] != 1 or max_read_error != 0 or max_identity_error >= 1e-5:
        raise AssertionError("Read-only or k=197 identity validation failed")
    logging.info("Identity passed: %s", checks)
    if config.get("purpose") == "identity_200":
        logging.info("200-image prerequisite complete; no sweep requested")
        return
    results = []
    completed = set()
    previous_predictions = None
    if resume_from is not None:
        previous_predictions = pd.read_csv(resume_from / 'predictions.csv')
        results = pd.read_csv(resume_from / 'results.csv').to_dict('records')
        completed = validate_completed_settings(previous_predictions, results, pd.DataFrame(identity_rows), config)
        logging.info('Validated and retained %d completed settings from %s', len(completed), resume_from)
        write_csv(output / 'results.csv', results)
    count_histograms = {}
    fields = ["method", "k", "seed", "index", "target", "original", "executed", "agree",
              "kl", "original_correct", "executed_correct"]
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        if previous_predictions is not None:
            writer.writerows(previous_predictions.to_dict('records'))
            stream.flush()
        with torch.inference_mode():
            for method, k, seed in settings(config):
                if (method, k, seed) in completed:
                    continue
                setting_start = time.perf_counter()
                generator = torch.Generator(device=device).manual_seed(seed) if method == "random" else None
                grouping, readset = ("kmeans" if method == "kmeans" else "grid"), ("random" if method == "random" else "ours")
                rows, offset = [], 0
                for batch_index, (images, labels) in enumerate(loader):
                    images = images.to(device)
                    counts = {} if method == "kmeans" and config.get("collect_grouping_counts") else None
                    with read_attention(model, k_top=k, grouping=grouping, readset=readset,
                                        generator=generator, grouping_counts=counts) as patterns:
                        original = model(images)
                    if counts is not None:
                        for layer, kinds in counts.items():
                            for kind, values in kinds.items():
                                key = (layer, kind)
                                histogram = torch.bincount(values.flatten(), minlength=198).cpu()
                                count_histograms[key] = count_histograms.get(key, torch.zeros(198, dtype=torch.long)) + histogram
                    if not torch.equal(original.cpu(), original_batches[batch_index]):
                        raise AssertionError("Reading or preprocessing changed original logits")
                    with execute_attention(model, patterns):
                        executed = model(images)
                    if not torch.isfinite(executed).all():
                        raise FloatingPointError("Non-finite execution output")
                    divergences = kl_divergence(original.double(), executed.double()).tolist()
                    agreements = agreement(original, executed).tolist()
                    for index, label, pred, exe, agree, kl in zip(
                        indices[offset:offset + len(images)], labels.tolist(), original.argmax(-1).tolist(),
                        executed.argmax(-1).tolist(), agreements, divergences,
                    ):
                        rows.append(dict(zip(fields, [method, k, seed, index, label, pred, exe, int(agree),
                                                     kl, int(pred == label), int(exe == label)])))
                    offset += len(images)
                writer.writerows(rows)
                stream.flush()
                frame = pd.DataFrame(rows)
                low, high = mean_interval(frame.agree.to_numpy(), config)
                results.append({
                    "run_id": output.name, "timestamp": timestamp, "git_commit": commit,
                    "model": config["model"], "dataset": config["dataset"], "n_images": len(indices),
                    "rq": "rq1", "replace_scope": "all", "layer": "", "grouping": grouping,
                    "readset": readset, "seed": seed, "mode": "strict", "m": 16, "k": k,
                    "agree": frame.agree.mean(), "agree_ci_low": low, "agree_ci_high": high,
                    "kl": frame.kl.mean(), "acc_original": frame.original_correct.mean(),
                    "acc_executed": frame.executed_correct.mean(),
                    "notes": f"frozen original summaries; seconds={time.perf_counter() - setting_start:.3f}",
                })
                write_csv(output / "results.csv", results)
                logging.info("%s k=%d seed=%d: agree=%.4f KL=%.4f; %.1fs", method, k, seed,
                             results[-1]["agree"], results[-1]["kl"], time.perf_counter() - setting_start)
    summary, contrasts = aggregate_predictions(pd.read_csv(output / "predictions.csv"), config)
    write_csv(output / "summary.csv", summary)
    write_csv(output / "contrasts.csv", contrasts)
    if count_histograms:
        write_csv(output / "group_query_counts.csv", [
            {"layer": layer, "kind": kind, "query_count": count, "frequency": int(frequency)}
            for (layer, kind), histogram in count_histograms.items()
            for count, frequency in enumerate(histogram.tolist())
        ])
    checks.update({"settings_completed": len(results), "settings_reused": len(completed),
                   "elapsed_seconds": time.perf_counter() - started})
    write_json(output / "checks.json", checks)
    logging.info("RQ1 complete: %d settings, %.1f seconds", len(results), checks["elapsed_seconds"])


if __name__ == "__main__":
    main()
