"""RQ3 human alignment (E7): do the patch maps agree with where people find the class evidence?

ClickMe validation set (ImageNet validation images with human maps; Fel et al. 2022 protocol, see
fastweight_vit.data.clickme). The methods are the explanations list (without the object mask), plus the mean
human map of the evaluated images (Fel et al.'s centre-bias control) and, when `layerwise` is set, single-layer
maps of each family in `layerwise_families` (default: the probe only). Per image and method:
- spearman_token: Spearman between the 196 patch scores and the human map averaged per patch;
- spearman_pixel: Spearman between the map upsampled (bilinear) to 224 x 224 and the human map; this is
  the Harmonization metric (divide by HUMAN_CEILING for their 'alignment');
- *_shuffled: the same against another image's human map (the next image in the batch). A fixed prior
  such as the centre scores the same on both, so own - shuffled is the image-specific part.
Output alignment.csv, with the model's top-1 and the label for stratification.
"""

import logging
import time

import torch
from timm.data import resolve_model_data_config
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from ..data.clickme import ClickMe
from ..io import write_json
from ..metrics import spearman
from ..models import IMAGENET_PREPROCESSING, data_reference, load_model
from ..execution import resolve
from ..runtime import data_root
from . import common
from .common import Sink
from .explanations import layerwise_methods, method_scores

MEASURES = ['spearman_token', 'spearman_token_shuffled', 'spearman_pixel', 'spearman_pixel_shuffled']


class _Heatmaps(Dataset):
    def __init__(self, clickme):
        self.clickme = clickme

    def __len__(self):
        return len(self.clickme)

    def __getitem__(self, index):
        return self.clickme.heatmap(index)


def mean_human_map(dataset, indices, config):
    total = torch.zeros(224, 224, dtype=torch.float64)
    for maps in DataLoader(Subset(_Heatmaps(dataset), indices), batch_size=config['batch_size'],
                           num_workers=config['num_workers']):
        total += maps.double().sum(0)
    return (total / len(indices)).float()


def per_patch(maps):
    """[B, 224, 224] -> [B, 196] patch means."""
    return F.avg_pool2d(maps[:, None], 16).flatten(1)


def upsample(scores):
    """[B, 196] -> [B, 224, 224], bilinear."""
    return F.interpolate(scores.view(-1, 1, 14, 14), size=(224, 224), mode='bilinear', align_corners=False)[:, 0]


def alignment(scores, human):
    """Dict of [B] tensors; the shuffled control pairs each image with the next one's human map."""
    other = human.roll(1, 0)
    pixel = upsample(scores)
    out = dict(spearman_token=spearman(scores, per_patch(human)), spearman_pixel=spearman(pixel, human),
               spearman_token_shuffled=spearman(scores, per_patch(other)),
               spearman_pixel_shuffled=spearman(pixel, other))
    if len(scores) == 1:
        for key in ('spearman_token_shuffled', 'spearman_pixel_shuffled'):
            out[key] = torch.full_like(out[key], float('nan'))
    return out


def run(args, config, output, commit):
    """args: model, smoke, resume. Methods and probe settings come from config['explanations']."""
    settings, explain_settings = config['human_alignment'], config['explanations']
    common.deterministic()
    device = common.DEVICE
    assert device != 'cuda' or torch.cuda.is_available(), 'Formal inference requires a GPU'
    model = load_model(args.model, device)
    pre = IMAGENET_PREPROCESSING[args.model] or resolve_model_data_config(data_reference(model))
    dataset = ClickMe(data_root() / 'clickme', pre['mean'], pre['std'], expected_shards=settings['shards'])
    indices = list(range(len(dataset)))
    indices = indices[:args.smoke] if args.smoke else indices
    methods = [entry for entry in explain_settings['methods'] if entry['method'] != 'mask']
    if settings['layerwise']:
        methods += layerwise_methods(len(model.blocks), settings.get('layerwise_families', ['probe']))
    spec = resolve(explain_settings['probe_spec'], config['main'])
    progress_path, done = common.progress(output, args.resume)
    write_json(output / f'metadata{"_resume" if done else ""}.json', common.provenance(
        commit, model=args.model, dataset='clickme_val', images=len(indices), records=len(dataset),
        methods=[entry['name'] for entry in methods] + ['mean_human'], probe_spec=spec, smoke_images=args.smoke,
        resumed_from=done))
    started = time.perf_counter()
    mean_map = mean_human_map(dataset, indices, config).to(device)
    torch.save(mean_map.cpu(), output / 'mean_human_map.pt')
    batches = common.make_batches(config, done, progress_path, started)
    sink = Sink(output / 'alignment.csv', ['model', 'index', 'label', 'predicted', 'method'] + MEASURES)
    for start, ids, (images, human, labels, _) in batches(f'clickme_{args.model}', dataset, indices):
        images, human = images.to(device), human.to(device)
        with torch.no_grad():
            predicted = model(images).argmax(-1).tolist()
        noise = torch.Generator(device=device).manual_seed(settings['random_seed'] * 1_000_003 + start)
        maps = {entry['name']: method_scores(entry, model, images, None, spec, noise).float() for entry in methods}
        maps['mean_human'] = per_patch(mean_map[None]).expand(len(ids), -1)
        rows = []
        for name, scores in maps.items():
            values = {key: value.tolist() for key, value in alignment(scores, human).items()}
            rows += [dict(model=args.model, index=index, label=int(labels[n]), predicted=predicted[n], method=name,
                          **{key: values[key][n] for key in MEASURES}) for n, index in enumerate(ids)]
        sink.write(rows)
    common.finish(output, started)
    logging.info('Complete: ClickMe alignment for %s', args.model)
