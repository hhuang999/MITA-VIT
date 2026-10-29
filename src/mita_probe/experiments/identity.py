"""Premise check for "each neuron is an image patch": does token j still stand for patch j at each layer?

On the ImageNet subset: tokens of the first `train_images` images train one linear position classifier per
layer, the next `test_images` images test it (chance 1/196); self-similarity compares each token with the
same position at layer 1. Output identity.csv: one row per layer.
"""

import logging
import time

import torch
from torch.utils.data import DataLoader, Subset

from ..identity import collect, self_similarity, train_position_probe
from ..io import write_csv, write_json
from ..models import load_model
from . import common, datasets


def run(args, config, output, commit):
    settings = config['identity']
    common.deterministic()
    device = common.DEVICE
    model = load_model(args.model, device)
    dataset, indices = datasets.build('imagenet', model, args.model, config)
    n_train, n_test = settings['train_images'], settings['test_images']
    if args.smoke:
        n_train, n_test = max(args.smoke - 2, 1), 2
    layers = list(range(1, len(model.blocks) + 1))
    write_json(output / 'metadata.json', common.provenance(commit, model=args.model, train_images=n_train,
                                                           test_images=n_test, epochs=settings['epochs']))
    started = time.perf_counter()
    features = {layer: [] for layer in layers}
    same, other = {layer: [] for layer in layers}, {layer: [] for layer in layers}
    loader = DataLoader(Subset(dataset, indices[:n_train + n_test]), batch_size=config['batch_size'],
                        num_workers=config['num_workers'], shuffle=False)
    for images, _ in loader:
        tokens = collect(model, images.to(device), layers)
        for layer in layers:
            features[layer].append(tokens[layer].half().cpu())
            a, b = self_similarity(tokens[layer], tokens[1])
            same[layer].append(a.cpu())
            other[layer].append(b.cpu())
    rows = []
    for layer in layers:
        stacked = torch.cat(features[layer])
        accuracy, distance = train_position_probe(stacked[:n_train].to(device), stacked[n_train:].to(device),
                                                  epochs=settings['epochs'], seed=settings['seed'])
        rows.append(dict(model=args.model, layer=layer, position_accuracy=accuracy, error_distance=distance,
                         self_cosine=float(torch.cat(same[layer]).mean()),
                         other_cosine=float(torch.cat(other[layer]).mean())))
        logging.info('layer %d: position accuracy %.3f', layer, accuracy)
    write_csv(output / 'identity.csv', rows)
    common.finish(output, started)
