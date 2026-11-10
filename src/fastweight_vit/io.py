"""Configuration and provenance for reproducible experiments."""

import csv
import json
import re
import subprocess
from pathlib import Path

import yaml


def merge(base: dict, override: dict) -> dict:
    """Recursive dict merge: nested sections are merged key by key, everything else is replaced."""
    out = dict(base)
    for key, value in override.items():
        out[key] = merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def load_config(path: Path) -> dict:
    """Read a YAML config; `base: other.yaml` inherits that file and overrides it section by section."""
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    base = config.pop("base", None)
    return merge(load_config(path.parent / base), config) if base else config


def source_commit(root: Path) -> str:
    if not (root / '.git').exists():
        marker = root / '.git_archival.txt'
        revision = marker.read_text(encoding='utf-8').strip() if marker.exists() else ''
        if not re.fullmatch(r'[0-9a-f]{40}', revision):
            raise RuntimeError('Run from a Git checkout or a git archive export with a source revision')
        return revision
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True
    )
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "src", "scripts",
         "configs", "tests", "reproduce"], cwd=root, text=True,
    )
    ignored = subprocess.check_output(
        ["git", "ls-files", "--others", "--ignored", "--exclude-standard", "src", "scripts",
         "configs", "tests", "reproduce"], cwd=root, text=True,
    )
    ignored = [path for path in ignored.splitlines()
               if path.endswith((".py", ".yaml", ".sh")) and "__pycache__" not in path]
    if status or untracked or ignored:
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
