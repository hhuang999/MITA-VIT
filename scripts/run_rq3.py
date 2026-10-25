"""RQ3: observe spatial reading patterns on all Oxford-IIIT Pet test images."""

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from mita_probe.runtime import configure_runtime, data_root

ROOT = configure_runtime()

import numpy as np
import pandas as pd
import timm
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from mita_probe.assets import MODEL_REVISION
from mita_probe.io import load_config, source_commit, write_csv, write_json
from mita_probe.metrics import foreground_ratio, heatmap_metrics
from mita_probe.models import capture_cls_attention, load_model, read_attention
from mita_probe.pet import PetSegmentation, background_regions, selection_counts
from mita_probe.rq1 import mean_interval


def interval(values, config):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan"), float("nan"), 0
    low, high = mean_interval(values, config)
    return float(values.mean()), low, high, len(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/rq3.yaml")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if (config["m"] != 16 or config["k"] != [16, 32] or config["mode"] != "observe"
            or config["heatmap_layers"] != [9, 10, 11, 12] or config["foreground_threshold"] != .5
            or config["gaussian_sigma_fraction"] != .25 or config["grouping"] != "grid"):
        raise ValueError("Retain the authorized RQ3 settings")
    commit = source_commit(ROOT)
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[
        logging.FileHandler(output / "log.txt", encoding="utf-8"), logging.StreamHandler()])
    try:
        run(config, output, commit)
    except Exception:
        logging.exception("RQ3 failed")
        raise


def run(config, output, commit):
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    device = config["device"]
    model = load_model(device)
    dataset = PetSegmentation(data_root(), model)
    indices = list(range(len(dataset)))
    if config.get("n_images"):
        indices = indices[:config["n_images"]]
    example_ids = np.random.default_rng(0).choice(len(dataset), 5, replace=False).tolist()[:3]
    write_json(output / "indices.json", indices)
    write_json(output / "metadata.json", {
        "timestamp": datetime.now(ZoneInfo("Asia/Singapore")).isoformat(), "git_commit": commit,
        "torch": torch.__version__, "timm": timm.__version__, "device": torch.cuda.get_device_name(),
        "checkpoint_revision": MODEL_REVISION, "preprocessing": dataset.preprocessing,
        "mask_transform": "same Resize(248)/CenterCrop(224), nearest interpolation",
        "example_indices": example_ids, "examples_selected_before_results": True,
        "trimap": {"1": "foreground", "2": "background", "3": "unclassified boundary"},
        "token_rule": "foreground >= half of nonboundary pixels; all-boundary excluded",
        "aggregation": "per-image ratios then mean; images are bootstrap units",
        "pointing_ties": "first row-major maximum among valid tokens",
        "source_regions": "adaptive-pooling windows, background if fewer than half valid tokens foreground",
    })
    loader = DataLoader(Subset(dataset, indices), batch_size=config["batch_size"], shuffle=False,
                        num_workers=config["num_workers"])
    coordinate = torch.arange(14, device=device, dtype=torch.float64) - 6.5
    gaussian = torch.exp(-(coordinate[:, None]**2 + coordinate[None, :]**2)/(2*3.5**2)).flatten().float()
    layer_rows, heatmap_rows, background_rows, observe_rows, example_rows = [], [], [], [], []
    started = time.perf_counter()
    max_observe_error = 0.
    with torch.inference_mode():
        for images, foreground, valid, batch_ids in loader:
            images, foreground, valid = images.to(device), foreground.to(device), valid.to(device)
            original = model(images)
            bg_regions = background_regions(foreground, valid)
            area = (foreground & valid).sum(1) / valid.sum(1)
            for k in config["k"]:
                with read_attention(model, k_top=k) as patterns, capture_cls_attention(model) as cls:
                    observed = model(images)
                error = (observed-original).abs().max().item()
                max_observe_error = max(max_observe_error, error)
                if error != 0 or not torch.isfinite(observed).all():
                    raise AssertionError("RQ3 observe mode changed original logits")
                for index, pred, obs in zip(batch_ids.tolist(), original.argmax(-1).tolist(), observed.argmax(-1).tolist()):
                    observe_rows.append({"index": index, "k": k, "original": pred, "observed": obs,
                                         "agree": int(pred == obs), "batch_max_logit_error": error})
                selection_maps = []
                for layer, pattern in patterns.items():
                    ratio = foreground_ratio(pattern.readsets, foreground, valid)
                    for index, value, fg_area in zip(batch_ids.tolist(), ratio.tolist(), area.tolist()):
                        layer_rows.append({"index": index, "k": k, "layer": layer,
                                           "fg_ratio": value, "fg_area_ratio": fg_area})
                    if layer in config["heatmap_layers"]:
                        selection_maps.append(selection_counts(pattern.readsets))
                        bg_ratio = foreground_ratio(pattern.readsets, foreground, valid, bg_regions)
                        for index, value, n_regions in zip(batch_ids.tolist(), bg_ratio.tolist(), bg_regions.sum(1).tolist()):
                            background_rows.append({"index": index, "k": k, "layer": layer,
                                                    "fg_ratio": value, "n_background_regions": n_regions})
                selected = torch.stack(selection_maps).mean(0)
                maps = {"selection": selected, "cls_attention": cls["heatmap"],
                        "center_gaussian": gaussian.expand(len(images), -1)}
                for method, heatmap in maps.items():
                    mass, hit = heatmap_metrics(heatmap, foreground, valid)
                    for index, mass_value, hit_value in zip(batch_ids.tolist(), mass.tolist(), hit.tolist()):
                        heatmap_rows.append({"index": index, "k": k, "method": method,
                                             "fg_mass": mass_value, "pointing": hit_value})
                if k == config["primary_k"]:
                    for position, index in enumerate(batch_ids.tolist()):
                        if index not in example_ids:
                            continue
                        _, rgb, _, _, _ = dataset.aligned(index)
                        rgb.save(output / f"example_{index}.png")
                        possible = bg_regions[position].nonzero().flatten()
                        region = int(possible[0]) if len(possible) else -1
                        background_map = torch.zeros(196, device=device)
                        if region >= 0:
                            # Fixed first background window; sum its six heads in layer 12.
                            background_map = selection_counts(patterns[12].readsets[position:position+1, :, region:region+1])[0]
                        for token in range(196):
                            example_rows.append({"index": index, "token": token,
                                "selection": float(selected[position, token]),
                                "cls_attention": float(cls["heatmap"][position, token]),
                                "background_reads": float(background_map[token]), "source_region": region,
                                "foreground": int(foreground[position, token]), "valid": int(valid[position, token])})
            if len(observe_rows) % (config["batch_size"]*20) == 0 or int(batch_ids[-1]) == indices[-1]:
                logging.info("Observed %d/%d images, both k; %.1fs", int(batch_ids[-1])+1, len(indices), time.perf_counter()-started)
    write_csv(output / "layer_predictions.csv", layer_rows)
    write_csv(output / "heatmap_predictions.csv", heatmap_rows)
    write_csv(output / "background_predictions.csv", background_rows)
    write_csv(output / "observe_predictions.csv", observe_rows)
    if example_rows:
        write_csv(output / "example_maps.csv", example_rows)
    layers, heatmaps, backgrounds = pd.DataFrame(layer_rows), pd.DataFrame(heatmap_rows), pd.DataFrame(background_rows)
    results = []
    for (k, layer), frame in layers.groupby(["k", "layer"]):
        mean, low, high, n = interval(frame.fg_ratio, config)
        area, area_low, area_high, _ = interval(frame.fg_area_ratio, config)
        results.append({"run_id": output.name, "method": "grid", "k": int(k), "layer": int(layer),
                        "fg_ratio": mean, "fg_ci_low": low, "fg_ci_high": high,
                        "fg_area_ratio": area, "area_ci_low": area_low, "area_ci_high": area_high,
                        "n_images": n})
    write_csv(output / "results.csv", results)
    heatmap_summary = []
    for (k, method), frame in heatmaps.groupby(["k", "method"]):
        mass, ml, mh, n = interval(frame.fg_mass, config)
        hit, hl, hh, _ = interval(frame.pointing, config)
        heatmap_summary.append({"k": int(k), "method": method, "fg_mass": mass,
            "mass_ci_low": ml, "mass_ci_high": mh, "pointing": hit,
            "pointing_ci_low": hl, "pointing_ci_high": hh, "n_images": n})
    write_csv(output / "heatmap_summary.csv", heatmap_summary)
    background_summary = []
    for k, frame in backgrounds.groupby("k"):
        for layer in [9, 10, 11, 12, "9-12"]:
            values = (frame.groupby("index").fg_ratio.mean() if layer == "9-12"
                      else frame[frame.layer == layer].fg_ratio)
            mean, low, high, n = interval(values, config)
            background_summary.append({"k": int(k), "layer": layer, "fg_ratio": mean,
                "ci_low": low, "ci_high": high, "n_images": n,
                "aggregation": "mean per-image ratio, then mean across layers for 9-12"})
    write_csv(output / "background_summary.csv", background_summary)
    write_json(output / "checks.json", {"n_images": len(indices), "max_observe_logit_error": max_observe_error,
        "observe_agreement": float(np.mean([r["agree"] for r in observe_rows])),
        "n_layer_records": len(layer_rows), "n_heatmap_records": len(heatmap_rows),
        "n_background_records": len(background_rows),
        "undefined_background_records": int(backgrounds.fg_ratio.isna().sum()),
        "undefined_heatmap_records": int(heatmaps.fg_mass.isna().sum()),
        "elapsed_seconds": time.perf_counter()-started})
    logging.info("RQ3 complete: %d images, %.1fs", len(indices), time.perf_counter()-started)


if __name__ == "__main__":
    main()
