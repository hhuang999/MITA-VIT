"""Invoke the installed WSL Colab CLI while redacting credential-bearing URLs."""

import re
import subprocess
import sys


def redact(text):
    text = re.sub(r"(https?://[^\s?'\"]+)\?[^\s'\"]+", r'\1?[REDACTED_QUERY]', text)
    text = re.sub(r'hf_[A-Za-z0-9]{20,}', '[REDACTED_HF_TOKEN]', text)
    return re.sub(r'Bearer\s+[^\s\"\']+', 'Bearer [REDACTED]', text)


if __name__ == '__main__':
    command = ['wsl', '-d', 'Ubuntu', '--exec',
               '/home/hhuan/.local/share/google-colab-cli-venv/bin/colab',
               '--auth', 'oauth2', *sys.argv[1:]]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding='utf-8', errors='replace', bufsize=1)
    for line in process.stdout:
        print(redact(line), end='', flush=True)
    raise SystemExit(process.wait())
