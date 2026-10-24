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
