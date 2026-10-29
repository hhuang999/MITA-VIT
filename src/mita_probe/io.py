"""Configuration and provenance for reproducible experiments."""

import csv
import json
import subprocess
from pathlib import Path

import yaml


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    base = config.pop("base", None)
    return (load_config(path.parent / base) if base else {}) | config


def source_commit(root: Path) -> str:
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True
    )
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "src", "scripts",
         "configs", "tests", "colab"], cwd=root, text=True,
    )
    if status or untracked:
        raise RuntimeError("Commit source and tracked changes before running experiments")
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
