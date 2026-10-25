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
    """Allow a persistent data volume without moving the repository."""
    configure_runtime()
    return Path(os.environ.get("MITA_DATA_ROOT", ROOT / "data")).resolve()


def project_path(path: str) -> Path:
    relative = Path(path)
    if not relative.is_absolute() and relative.parts[0] == "data":
        return data_root().joinpath(*relative.parts[1:])
    return ROOT / relative
