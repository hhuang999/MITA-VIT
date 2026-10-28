"""Post-hoc (registered after seeing V2 Pet results): union-based composition of reading sets.

MiTA's token-pruning figure shows tokens read by at least one set. The preregistered
composition counts selections with multiplicity, which over-weights tokens many sets
share. This stage counts each token once per layer, over all regions and heads.
"""

import argparse
import csv
import hashlib
from pathlib import Path

from mita_probe.runtime import configure_runtime, data_root, project_path
ROOT = configure_runtime()

import torch
from torch.utils.data import DataLoader

from mita_probe.anatomy import capture_qk
from mita_probe.io import load_config, source_commit, write_json
from mita_probe.mita_model import CHECKPOINT_SHA256, load_mita_model
from mita_probe.models import load_model
from mita_probe.pet import PetSegmentation
from mita_probe.rq3_extension import capture_input_norms, sink_tokens
from mita_probe.settings import base_pattern, resolve

FIELDS = ['model', 'index', 'spec', 'layer', 'union_size', 'sink', 'foreground', 'background', 'unknown', 'fg_area']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', type=int, default=0)
    args = parser.parse_args()
    config = load_config(ROOT / 'configs/anatomy_v2.yaml')
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=False)
    commit = source_commit(ROOT)
    torch.use_deterministic_algorithms(True)
    deit = load_model('cuda')
    path = project_path(config['mita']['checkpoint'])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == CHECKPOINT_SHA256
    mita = load_mita_model(path, 'cuda')
    dataset = PetSegmentation(data_root(), deit)
    n = args.smoke or len(dataset)
    loader = DataLoader(torch.utils.data.Subset(dataset, range(n)), batch_size=32, shuffle=False)
    write_json(output / 'metadata.json', dict(git_commit=commit, post_hoc=True, images=n,
                                              device=torch.cuda.get_device_name()))
    with (output / 'union_composition.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for kind, model in [('deit', deit), ('mita', mita)]:
            scale, grid = model.blocks[0].attn.scale, model.patch_embed.grid_size
            with torch.inference_mode():
                for images, fg, valid, ids in loader:
                    images, fg, valid = images.cuda(), fg.cuda(), valid.cuda()
                    with capture_qk(model) as qk, capture_input_norms(model) as norms:
                        model(images)
                    fg_area = ((fg & valid).sum(-1) / valid.sum(-1).clamp_min(1)).tolist()
                    rows = []
                    for layer in range(1, 13):
                        sink = sink_tokens(norms[layer], config['pet']['sink_threshold'])
                        for comp in config['pet']['composition']:
                            spec = resolve(dict(mode='L3', routing='dot', cls='routed') | comp,
                                           dict(scoring='query', summary_scale='mita'))
                            pattern, _ = base_pattern(spec, *qk[layer], scale, grid)
                            used = torch.zeros(len(ids), 197, dtype=torch.bool, device='cuda')
                            used.scatter_(1, pattern.readsets.flatten(1), True)
                            used = used[:, 1:]  # spatial tokens only
                            size = used.sum(-1).clamp_min(1)
                            parts = dict(sink=used & sink, foreground=used & fg & valid & ~sink,
                                         background=used & ~fg & valid & ~sink, unknown=used & ~valid & ~sink)
                            fractions = {key: (value.sum(-1) / size).tolist() for key, value in parts.items()}
                            name = f"m{comp['m_side'] ** 2}_k{comp['k']}_{comp['landmark_mode']}"
                            for i, index in enumerate(ids.tolist()):
                                rows.append(dict(model=kind, index=index, spec=name, layer=layer,
                                                 union_size=int(used[i].sum()), fg_area=fg_area[i],
                                                 **{key: value[i] for key, value in fractions.items()}))
                    writer.writerows(rows)
    write_json(output / 'checks.json', dict(complete=True))


if __name__ == '__main__':
    main()
