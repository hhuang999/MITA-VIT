"""Inspect a Colab runtime without loading model weights or credentials."""

import json
import platform
import shutil
import subprocess
import sys

print(json.dumps({"python": sys.version, "platform": platform.platform(),
                  "disk_free_gib": shutil.disk_usage('/content').free / 2**30}, indent=2))
subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
                '--format=csv,noheader'], check=True)

