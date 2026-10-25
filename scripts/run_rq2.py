"""RQ2: replace exactly one DeiT layer at a time on the fixed ImageNet sample."""

import argparse
import csv
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
from mita_probe.data import ImageNetValidation, experiment_indices
from mita_probe.io import load_config, source_commit, write_csv, write_json
from mita_probe.metrics import kl_divergence
from mita_probe.models import execute_attention, load_model, read_attention
from mita_probe.rq1 import mean_interval


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/rq2.yaml")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if (config["m"] != 16 or config["k"] != [8, 32]
            or config["layers"] != list(range(1, 13)) or config["subset_seed"] != 0
            or config["grouping"] != "grid" or config["mode"] != "strict"):
        raise ValueError("RQ2 must retain the prespecified grid, m=16, k=8/32, 12 layers")
    commit = source_commit(ROOT)
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[
        logging.FileHandler(output / "log.txt", encoding="utf-8"), logging.StreamHandler()])
    try:
        run(config, output, commit)
    except Exception:
        logging.exception("RQ2 failed")
        raise


def run(config, output, commit):
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    model = load_model(config["device"])
    dataset = ImageNetValidation(data_root() / "imagenet_val", model)
    indices = experiment_indices(len(dataset), config["n_images"], 0, project_path(config["indices"]))
    write_json(output / "indices.json", indices)
    timestamp = datetime.now(ZoneInfo("Asia/Singapore")).isoformat()
    write_json(output / "metadata.json", {
        "timestamp": timestamp, "git_commit": commit, "torch": torch.__version__,
        "timm": timm.__version__, "device": torch.cuda.get_device_name(),
        "checkpoint_revision": MODEL_REVISION, "preprocessing": dataset.preprocessing,
        "dtype": "float32, no autocast", "layer_numbering": "CSV 1..12, model.blocks index 0..11",
        "explanation_source": "original-model summary; only one layer executed per pass",
    })
    loader = DataLoader(Subset(dataset, indices), batch_size=config["batch_size"],
                        shuffle=False, num_workers=config["num_workers"])
    started = time.perf_counter()
    identity_rows = []
    max_read_error = max_identity_error = 0.
    fields = ["layer", "layer_index", "k", "index", "target", "original", "executed",
              "agree", "kl", "original_correct", "executed_correct"]
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        with torch.inference_mode():
            offset = 0
            for images, labels in loader:
                images = images.to(config["device"])
                original = model(images)
                with read_attention(model, k_top=197) as identity_patterns:
                    observed = model(images)
                with execute_attention(model, identity_patterns):
                    identity = model(images)
                max_read_error = max(max_read_error, (original - observed).abs().max().item())
                max_identity_error = max(max_identity_error, (original - identity).abs().max().item())
                batch_ids = indices[offset:offset + len(images)]
                for index, label, pred, ident in zip(batch_ids, labels.tolist(),
                        original.argmax(-1).tolist(), identity.argmax(-1).tolist()):
                    identity_rows.append({"index": index, "target": label, "original": pred,
                                          "identity": ident, "agree": int(pred == ident)})
                if max_read_error != 0 or max_identity_error >= 1e-5:
                    raise AssertionError("Identity or observe invariance failed")
                for k in config["k"]:
                    with read_attention(model, k_top=k) as patterns:
                        observed = model(images)
                    if not torch.equal(observed, original):
                        raise AssertionError("Observing summaries changed original logits")
                    for layer in config["layers"]:
                        with execute_attention(model, {layer: patterns[layer]}):
                            executed = model(images)
                        if not torch.isfinite(executed).all():
                            raise FloatingPointError("Non-finite RQ2 output")
                        divergences = kl_divergence(original.double(), executed.double()).tolist()
                        rows = []
                        for index, label, pred, exe, kl in zip(batch_ids, labels.tolist(),
                                original.argmax(-1).tolist(), executed.argmax(-1).tolist(), divergences):
                            rows.append(dict(zip(fields, [layer, layer - 1, k, index, label, pred, exe,
                                int(pred == exe), kl, int(pred == label), int(exe == label)])))
                        writer.writerows(rows)
                stream.flush()
                offset += len(images)
                if offset % (config["batch_size"] * 10) == 0 or offset == len(indices):
                    logging.info("All 24 settings: %d/%d images; %.1fs", offset, len(indices), time.perf_counter()-started)
    write_csv(output / "identity_predictions.csv", identity_rows)
    frame = pd.read_csv(output / "predictions.csv")
    results = []
    for (k, layer), group in frame.groupby(["k", "layer"]):
        if len(group) != len(indices) or group["index"].duplicated().any():
            raise AssertionError("Incomplete or duplicate single-layer setting")
        low, high = mean_interval(group.agree.to_numpy(), config)
        kl_low, kl_high = mean_interval(group.kl.to_numpy(), config)
        results.append({"run_id": output.name, "timestamp": timestamp, "git_commit": commit,
            "model": config["model"], "dataset": config["dataset"], "n_images": len(indices),
            "rq": "rq2", "replace_scope": "layer", "layer": int(layer), "layer_index": int(layer)-1,
            "grouping": "grid", "readset": "ours", "seed": 0, "mode": "strict", "m": 16, "k": int(k),
            "agree": group.agree.mean(), "agree_ci_low": low, "agree_ci_high": high,
            "kl": group.kl.mean(), "kl_ci_low": kl_low, "kl_ci_high": kl_high,
            "acc_original": group.original_correct.mean(), "acc_executed": group.executed_correct.mean(),
            "notes": "one layer; original frozen summary equals online summary at this layer"})
    write_csv(output / "results.csv", results)
    checks = {"n_images": len(indices), "settings_completed": len(results),
        "identity_agreement": sum(x["agree"] for x in identity_rows)/len(indices),
        "max_read_logit_error": max_read_error, "max_identity_logit_error": max_identity_error,
        "elapsed_seconds": time.perf_counter()-started}
    write_json(output / "checks.json", checks)
    if checks["identity_agreement"] != 1:
        raise AssertionError("Identity agreement must be 100%")
    logging.info("RQ2 complete: %s", checks)


if __name__ == "__main__":
    main()
