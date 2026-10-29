"""Runs ON the Colab VM: gzip large CSVs, copy the run logs, and tar the result folders matching a prefix.

Usage inside an exec file: sys.argv = ['pack', '2026-11-02_']; exec(...)
"""
import subprocess
import sys

prefix = sys.argv[1]
command = (f"cd /content/MITA-VIT && mkdir -p experiments/{prefix}colab_env && "
           f"cp /content/run_env/* /content/*.log experiments/{prefix}colab_env/ 2>/dev/null; "
           f"find experiments/{prefix}* -name '*.csv' -size +5M -exec gzip -9 {{}} \\; ; "
           f"tar czf /content/results.tar.gz experiments/{prefix}* && ls -la /content/results.tar.gz")
print(subprocess.run(['bash', '-lc', command], capture_output=True, text=True).stdout)
