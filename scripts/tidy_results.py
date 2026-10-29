"""Prepare run folders for Git: score arrays and CSVs over 20 MB go gzipped into raw/ (ignored, kept on disk),
CSVs over 5 MB are gzipped in place. Analysis scripts read either form (mita_probe.results.load).

Usage: python scripts/tidy_results.py RUN [RUN ...]
"""

import argparse
import gzip
import shutil
from pathlib import Path

MB = 1 << 20


def gzip_to(source: Path, target: Path):
    with source.open('rb') as stream, gzip.open(target, 'wb', compresslevel=6) as out:
        shutil.copyfileobj(stream, out, 1 << 24)
    source.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path, nargs='+')
    args = parser.parse_args()
    for run in args.runs:
        raw = run / 'raw'
        if (run / 'scores').exists():
            raw.mkdir(exist_ok=True)
            shutil.move(str(run / 'scores'), str(raw / 'scores'))
        for path in sorted(run.glob('*.csv')):
            size = path.stat().st_size
            if size > 20 * MB:
                raw.mkdir(exist_ok=True)
                gzip_to(path, raw / f'{path.name}.gz')
            elif size > 5 * MB:
                gzip_to(path, path.with_name(f'{path.name}.gz'))
        print(run.name, sorted(p.name for p in run.iterdir()))


if __name__ == '__main__':
    main()
