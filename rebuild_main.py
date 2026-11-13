import os
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
display = root / "scripts" / "display"
os.environ.setdefault("SOURCE_DATE_EPOCH", "1794528000")
sys.path.insert(0, str(display))
from build_static_figures import fast_weight_lens

if __name__ == "__main__":
    (root / "paper" / "figures").mkdir(parents=True, exist_ok=True)
    fast_weight_lens()
    subprocess.run([sys.executable, str(display / "build_diagnostic_figure.py")], cwd=root, check=True)
    subprocess.run([sys.executable, str(display / "build_exploratory_tables.py"), "--main-only"], cwd=root, check=True)
