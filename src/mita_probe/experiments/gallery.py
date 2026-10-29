"""Qualitative examples for the figures: every method's map and the probe at each layer on random images.

For each dataset in `datasets`, `images` images are drawn with subset_seed (not hand-picked). Saved as
gallery_<dataset>.npz: index, rgb (uint8 [N, 224, 224, 3], the model's input crop), truth (object patches
[N, 196] for mask datasets, the human map [N, 224, 224] for ClickMe, part ids [N, 224, 224] for
PartImageNet), top1, one [N, 196] array per method and probe_layers [N, depth, 196].
"""

import time

import numpy as np
import torch
from timm.data import resolve_model_data_config

from ..data.clickme import ClickMe
from ..io import write_json
from ..models import IMAGENET_PREPROCESSING, data_reference, load_model
from ..probe import resolve
from ..runtime import data_root
from . import common, datasets
from .explanations import layerwise_methods, method_scores


def examples(name, model, model_name, config, indices):
    """(input tensors, rgb uint8, truth) for the chosen indices of one dataset."""
    if name == 'clickme':
        pre = IMAGENET_PREPROCESSING[model_name] or resolve_model_data_config(data_reference(model))
        clickme = ClickMe(data_root() / 'clickme', pre['mean'], pre['std'],
                          expected_shards=config['human_alignment']['shards'])
        tensors, rgbs, truths = [], [], []
        for index in indices:
            image, _, _ = clickme.record(index)
            tensors.append(clickme.image_tensor(image))
            rgbs.append(np.array(image.resize((224, 224))))
            truths.append(clickme.heatmap(index).numpy())
        return torch.stack(tensors), np.stack(rgbs), np.stack(truths)
    dataset, _ = datasets.build(name, model, model_name, config)
    tensors, rgbs, truths = [], [], []
    for index in indices:
        tensor, rgb, _, foreground, _ = dataset.aligned(index)
        tensors.append(tensor)
        rgbs.append(np.array(rgb))
        truths.append(dataset.cropped_labels(index).numpy() if name == 'partimagenet' else foreground.numpy())
    return torch.stack(tensors), np.stack(rgbs), np.stack(truths)


def dataset_size(name, model, model_name, config):
    if name == 'clickme':
        pre = IMAGENET_PREPROCESSING[model_name] or resolve_model_data_config(data_reference(model))
        return len(ClickMe(data_root() / 'clickme', pre['mean'], pre['std'],
                           expected_shards=config['human_alignment']['shards']))
    return len(datasets.build(name, model, model_name, config)[1])


def run(args, config, output, commit):
    """args: model. Settings in config['gallery']; methods from config['explanations']."""
    settings, explain_settings = config['gallery'], config['explanations']
    common.deterministic()
    device = common.DEVICE
    model = load_model(args.model, device)
    methods = [entry for entry in explain_settings['methods'] if entry['method'] != 'mask']
    layers = layerwise_methods(len(model.blocks))
    spec = resolve(explain_settings['probe_spec'], config['main'])
    started, chosen = time.perf_counter(), {}
    for name in settings['datasets']:
        rng = np.random.default_rng(settings['subset_seed'])
        size = dataset_size(name, model, args.model, config)
        indices = sorted(rng.choice(size, min(settings['images'], size), replace=False).tolist())
        chosen[name] = indices
        tensors, rgb, truth = examples(name, model, args.model, config, indices)
        images = tensors.to(device)
        with torch.no_grad():
            top1 = model(images).argmax(-1).cpu().numpy()
        noise = torch.Generator(device=device).manual_seed(settings['random_seed'])
        maps = {entry['name']: method_scores(entry, model, images, None, spec, noise).float().cpu().numpy()
                for entry in methods}
        per_layer = np.stack([method_scores(entry, model, images, None, spec, noise).float().cpu().numpy()
                              for entry in layers], 1)
        np.savez_compressed(output / f'gallery_{name}.npz', index=np.asarray(indices), rgb=rgb, truth=truth, top1=top1,
                            probe_layers=per_layer, **maps)
    write_json(output / 'metadata.json', common.provenance(commit, model=args.model, chosen=chosen,
                                                           methods=[entry['name'] for entry in methods]))
    common.finish(output, started)
