"""Configure local credentials and project-owned download caches."""

import os
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[2]


def configure_runtime() -> Path:
    load_dotenv(ROOT / ".env")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("HF_HOME", str(ROOT / "checkpoints" / "huggingface"))
    os.environ.setdefault("TORCH_HOME", str(ROOT / "checkpoints" / "torch"))
    return ROOT


def data_root() -> Path:
    """Datasets live under data/ unless FASTWEIGHT_DATA_ROOT points at another volume."""
    configure_runtime()
    return Path(os.environ.get("FASTWEIGHT_DATA_ROOT", ROOT / "data")).resolve()


def project_path(path: str) -> Path:
    """Repository path; data/... (except the fixed lists in data/splits) resolves under data_root()."""
    relative = Path(path)
    if not relative.is_absolute() and relative.parts[0] == "data" and relative.parts[1:2] != ("splits",):
        return data_root().joinpath(*relative.parts[1:])
    return ROOT / relative
