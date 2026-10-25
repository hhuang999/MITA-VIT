"""Save five fixed random Pet image/trimap/token overlays before RQ3."""

import argparse
import hashlib
from pathlib import Path

from mita_probe.runtime import configure_runtime, data_root

ROOT = configure_runtime()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from mita_probe.io import load_config, source_commit, write_json
from mita_probe.models import load_model
from mita_probe.pet import PetSegmentation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    commit = source_commit(ROOT)
    config = load_config(ROOT / "configs/rq3.yaml")
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=False)
    model = load_model(pretrained=False)
    data = PetSegmentation(data_root(), model)
    indices = np.random.default_rng(config["alignment_seed"]).choice(len(data), 5, replace=False).tolist()
    fig, axes = plt.subplots(5, 3, figsize=(9, 14), layout="constrained")
    checks = []
    for row, index in enumerate(indices):
        tensor, rgb, mask, foreground, valid = data.aligned(index)
        original, _ = data.dataset[index]
        torch.testing.assert_close(tensor, data.transform(original), rtol=0, atol=0)
        mask_rgba = np.zeros((224, 224, 4))
        mask_rgba[mask.numpy() == 1] = [0, .8, .3, .35]
        mask_rgba[mask.numpy() == 3] = [1, .2, .2, .55]
        token_rgba = np.zeros((14, 14, 4))
        token_rgba[foreground.reshape(14, 14).numpy()] = [0, .8, .3, .4]
        token_rgba[~valid.reshape(14, 14).numpy()] = [1, .2, .2, .6]
        for ax in axes[row]:
            ax.imshow(rgb)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row, 1].imshow(mask_rgba)
        axes[row, 2].imshow(token_rgba, extent=(-.5, 223.5, 223.5, -.5), interpolation="nearest")
        for boundary in np.arange(-.5, 224, 16):
            axes[row, 2].axvline(boundary, color="white", lw=.3, alpha=.6)
            axes[row, 2].axhline(boundary, color="white", lw=.3, alpha=.6)
        axes[row, 0].set_ylabel(f"Test index {index}")
        checks.append({"index": index, "mask_values": torch.unique(mask).tolist(),
                       "foreground_tokens": int(foreground.sum()), "valid_tokens": int(valid.sum())})
    for ax, title in zip(axes[0], ["Eval image", "Trimap: foreground / boundary", "14 x 14 foreground tokens"]):
        ax.set_title(title, fontsize=10)
    for extension in ["png", "pdf"]:
        fig.savefig(output / f"alignment.{extension}", dpi=160)
    plt.close(fig)
    readme = data_root() / "oxford-iiit-pet/annotations/README"
    write_json(output / "checks.json", {"git_commit": commit, "n_test_images": len(data),
        "indices": indices, "preprocessing": data.preprocessing,
        "transform": str(data.transform), "mask_interpolation": "nearest",
        "trimap_labels": {"1": "foreground", "2": "background", "3": "unclassified boundary"},
        "official_readme_sha256": hashlib.sha256(readme.read_bytes()).hexdigest(), "images": checks})
    (output / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (output / "log.txt").write_text("Verified original dataset README labels 1/2/3.\n"
        "Image and mask share Resize(size=248) then CenterCrop(224); mask uses nearest.\n"
        "Image tensors exactly match pinned timm eval transform. Visual review pending.\n", encoding="utf-8")
    print(indices)


if __name__ == "__main__":
    main()
