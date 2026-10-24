"""Validate original accuracy, read-only invariance, and full-key execution on Imagenette."""

import argparse
import csv
import json
import logging
import math
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from mita_probe.runtime import configure_runtime

ROOT = configure_runtime()

import torch
import yaml
from torch.utils.data import DataLoader, Subset

from mita_probe.data import Imagenette1k, subset_indices
from mita_probe.metrics import agreement, kl_divergence
from mita_probe.models import execute_attention, load_model, read_attention


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/validate_p1.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config["model"] != "deit_small_patch16_224.fb_in1k" or config["dataset"] != "imagenette":
        raise ValueError("P1 validation uses DeiT-S and Imagenette")
    if config["k"] != 197 or config["grouping"] != "grid" or config["readset"] != "ours":
        raise ValueError("This validation checks grid summaries with all 197 keys")
    if config["split"] != "val" or config["mode"] != "strict":
        raise ValueError("P1 validation requires validation images and strict execution")
    m_side = math.isqrt(config["m"])
    if m_side**2 != config["m"]:
        raise ValueError("m must be a square number of regions")
    output = ROOT / config["output"]
    output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(output / "log.txt", encoding="utf-8"), logging.StreamHandler()],
    )
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    diff = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True)
    if diff:
        raise RuntimeError("Commit tracked code and configuration before validation")
    torch.set_num_threads(2)
    torch.manual_seed(config["subset_seed"])
    torch.use_deterministic_algorithms(True)
    model = load_model(config["device"])
    dataset = Imagenette1k(ROOT / "data", model)
    indices = subset_indices(len(dataset), config["n_images"], config["subset_seed"])
    (output / "indices.json").write_text(json.dumps(indices) + "\n", encoding="utf-8")
    loader = DataLoader(
        Subset(dataset, indices), batch_size=config["batch_size"], shuffle=False,
        num_workers=config["num_workers"],
    )
    timestamp = datetime.now(ZoneInfo("Asia/Singapore")).isoformat()
    metadata = {
        "timestamp": timestamp, "git_commit": commit, "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0), "preprocessing": dataset.preprocessing,
        "checkpoint_revision": (ROOT / "checkpoints/huggingface/hub/models--timm--deit_small_patch16_224.fb_in1k/refs/main").read_text().strip(),
        "explanation_source": "original model, fixed for the execution pass",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    logging.info("Validating %d images at commit %s", len(indices), commit)
    correct = correct_executed = agreed = read_agreed = seen = 0
    max_read_error = max_execute_error = total_kl = 0.
    started = time.perf_counter()
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["index", "target", "original", "observed", "executed"])
        with torch.inference_mode():
            for images, labels in loader:
                images, labels = images.to(config["device"]), labels.to(config["device"])
                original = model(images)
                with read_attention(model, k_top=config["k"], m_side=m_side) as patterns:
                    observed = model(images)
                with execute_attention(model, patterns):
                    executed = model(images)
                if not all(torch.isfinite(x).all() for x in [original, observed, executed]):
                    raise FloatingPointError("Non-finite model output")
                batch = len(images)
                predictions = original.argmax(-1)
                executed_predictions = executed.argmax(-1)
                correct += predictions.eq(labels).sum().item()
                correct_executed += executed_predictions.eq(labels).sum().item()
                agreed += agreement(original, executed).sum().item()
                read_agreed += agreement(original, observed).sum().item()
                max_read_error = max(max_read_error, (observed - original).abs().max().item())
                max_execute_error = max(max_execute_error, (executed - original).abs().max().item())
                total_kl += kl_divergence(original.double(), executed.double()).sum().item()
                writer.writerows(zip(
                    indices[seen:seen + batch], labels.cpu().tolist(), predictions.cpu().tolist(),
                    observed.argmax(-1).cpu().tolist(), executed_predictions.cpu().tolist(),
                ))
                seen += batch
                if seen % (config["batch_size"] * 10) == 0 or seen == len(indices):
                    logging.info("%d/%d images; max execution logit error %.8g", seen, len(indices), max_execute_error)
    torch.cuda.synchronize()
    result = {
        "run_id": output.name, "timestamp": timestamp, "git_commit": commit,
        "model": config["model"], "dataset": "imagenette_val_160px", "n_images": seen,
        "rq": "p1_validation", "replace_scope": "all", "layer": "", "grouping": "grid",
        "readset": "ours", "seed": config["subset_seed"], "mode": "strict", "m": config["m"],
        "k": config["k"], "agree": agreed / seen, "agree_ci_low": "", "agree_ci_high": "",
        "kl": total_kl / seen, "acc_original": correct / seen, "acc_executed": correct_executed / seen,
        "notes": "P1 full-key identity check; not a fidelity sweep",
    }
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=result)
        writer.writeheader()
        writer.writerow(result)
    checks = {
        "original_correct": correct, "n_images": seen, "execution_agreed": agreed,
        "read_agreed": read_agreed, "max_read_logit_error": max_read_error,
        "max_execute_logit_error": max_execute_error,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "checks.json").write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    logging.info("Results: %s; checks: %s", result, checks)
    assert read_agreed == seen and max_read_error == 0, "Reading changed the original computation"
    assert agreed == seen, "Full-key execution changed predictions"
    assert max_execute_error < config["max_logit_error"], "Full-key error exceeds tolerance"
    logging.info("P1 validation passed")


if __name__ == "__main__":
    main()
