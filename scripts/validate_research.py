"""Audit preserved research observations and registered outcomes on CPU."""

import argparse
from datetime import datetime, timezone
import json
from importlib.metadata import version
from pathlib import Path
import platform
import sys
import subprocess
import tempfile

from fastweight_vit.evidence import rebuild, validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strict', action='store_true', help='check mandatory research evidence (the default)')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--rebuild', action='store_true', help='rebuild in a temporary directory and compare every table')
    parser.add_argument('--report', type=Path, help='save the actual dated audit report')
    args = parser.parse_args()
    report = {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'python': sys.version,
              'platform': platform.platform(), 'scope': 'CPU research artifact audit',
              'command': [sys.executable, *sys.argv],
              'source_revision': subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.root,text=True).strip()
                    if (args.root/'.git').exists() else (args.root/'.git_archival.txt').read_text().strip(),
              'packages': {name:version(name) for name in ('torch','torchvision','timm','numpy','pandas','matplotlib','pytest')}}
    try:
        report['integrity'] = validate(args.root)
        if args.rebuild:
            with tempfile.TemporaryDirectory(prefix='vit-rebuild-') as temp:
                report['rebuild'] = rebuild(args.root, Path(temp)/'build')
        report['passed'] = True
    except (ValueError, OSError, KeyError, EOFError, subprocess.CalledProcessError) as error:
        report.update(passed=False, error=str(error))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report if not report['passed'] else {k:v for k,v in report.items() if k != 'integrity'}, indent=2))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
