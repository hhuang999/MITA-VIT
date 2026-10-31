"""S3 (decision B24): what each single layer reads, and how much of it lands on high-norm tokens.

For each image of a mask dataset (ImageNet-S) and each layer l, three single-layer maps (the families of
explanations.layerwise_methods): the probe (attention mass under the reading sets), dense attention mass
and CLS attention. Per image, family and layer:
- high_fraction: share of patches that are high-norm at layer l, i.e. whose block-l input norm (before norm1)
  exceeds `norm_ratio` times the image's midpoint median patch norm at that layer (interventions.sink_tokens,
  the RQ3 diagnosis definition);
- high_share: share of the map's mass on those patches;
- mass and enrichment: object mass share and same-distance-ring enrichment (mita_probe.plausibility).
Output layer_reading.csv.
"""

import logging
import time

import torch

from .. import plausibility
from ..interventions import sink_tokens
from ..io import write_json
from ..models import load_model
from ..probe import capture_input_norms, resolve
from . import common, datasets
from .common import Sink
from .explanations import layerwise_methods, method_scores

FIELDS = ['model', 'index', 'family', 'layer', 'high_fraction', 'high_share', 'mass', 'enrichment']


def run(args, config, output, commit):
    """args: model, smoke, resume. Settings in config['layer_reading']."""
    settings = config['layer_reading']
    common.deterministic()
    device = common.DEVICE
    assert device != 'cuda' or torch.cuda.is_available(), 'Formal inference requires a GPU'
    model = load_model(args.model, device)
    dataset, indices = datasets.build(settings['dataset'], model, args.model, config)
    indices = indices[:args.smoke] if args.smoke else indices
    methods = layerwise_methods(len(model.blocks), settings['families'])
    spec = resolve(config['explanations']['probe_spec'], config['main'])
    progress_path, done = common.progress(output, args.resume)
    write_json(output / f'metadata{"_resume" if done else ""}.json', common.provenance(
        commit, model=args.model, dataset=settings['dataset'], images=len(indices), families=settings['families'],
        norm_ratio=settings['norm_ratio'], probe_spec=spec, smoke_images=args.smoke, resumed_from=done))
    started = time.perf_counter()
    batches = common.make_batches(config, done, progress_path, started)
    sink = Sink(output / 'layer_reading.csv', FIELDS)
    for start, ids, batch in batches(f'layers_{args.model}', dataset, indices):
        images, foreground, valid = (tensor.to(device) for tensor in batch[:3])
        with torch.no_grad(), capture_input_norms(model) as norms:
            model(images)
        high = {layer: sink_tokens(value, settings['norm_ratio']) for layer, value in norms.items()}
        other = foreground.roll(1, 0) if len(ids) > 1 else torch.zeros_like(foreground)
        rows = []
        for entry in methods:
            family, layer = entry['name'].rsplit('_layer', 1)
            layer = int(layer)
            scores = method_scores(entry, model, images, foreground, spec, None).float()
            positive = scores.clamp_min(0)
            share = ((positive * high[layer]).sum(1) / positive.sum(1)).tolist()
            fraction = high[layer].float().mean(1).tolist()
            values = plausibility.measures(scores, foreground, valid, other,
                                           generator=torch.Generator().manual_seed(start))
            rows += [dict(model=args.model, index=index, family=family, layer=layer, high_fraction=fraction[n],
                          high_share=share[n], mass=float(values['mass'][n]),
                          enrichment=float(values['enrichment'][n])) for n, index in enumerate(ids)]
        sink.write(rows)
    common.finish(output, started)
    logging.info('Complete: layer reading for %s', args.model)
