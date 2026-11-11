"""Export a committed full evidence bundle and an inventory with SHA-256 checksums."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

from fastweight_vit.io import source_commit
from fastweight_vit.release import archive_check, validate


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=root/'dist/course_code.zip')
    args = parser.parse_args()
    revision = source_commit(root)
    validate(root)
    archive_check(root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['git','archive','--format=zip','-o',str(args.output.resolve()),revision],cwd=root,check=True)
    with zipfile.ZipFile(args.output) as z:
        inventory = [{'path':name,'bytes':z.getinfo(name).file_size,
                      'sha256':hashlib.sha256(z.read(name)).hexdigest()} for name in z.namelist() if not name.endswith('/')]
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    report = {'source_revision':revision,'paper_version':8,'bundle':'full evidence; external datasets/checkpoints excluded',
              'zip_sha256':digest,'zip_bytes':args.output.stat().st_size,'files':inventory}
    args.output.with_suffix('.manifest.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    args.output.with_suffix('.zip.sha256').write_text(f'{digest}  {args.output.name}\n',encoding='utf-8')
    print(f'{len(inventory)} files; source {revision}; SHA-256 {digest}')


if __name__ == '__main__':
    main()
