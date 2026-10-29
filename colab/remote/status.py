"""Runs ON the Colab VM: print the job status file and whether the queue is still running."""
import subprocess

print(subprocess.run(['bash', '-lc', 'cat /content/run_env/run_status.txt 2>/dev/null; '
                      'ps aux | grep "[j]obs.sh" >/dev/null && echo RUNNING || echo NOT_RUNNING'],
                     capture_output=True, text=True).stdout)
