"""Does token j still stand for image patch j in layer l? (the premise of "each neuron is a patch")

Deep tokens mix information from the whole image (Abnar & Zuidema 2020; Meister et al. 2021), and
high-norm tokens lose their position (Darcet et al. 2024). Two checks per layer, on the tokens each
attention layer reads (the normalised block input):

- position: a linear classifier trained on some images to predict a token's grid position (196 classes),
  tested on held-out images; chance is 1/196. High accuracy means the token still knows where it is.
- self-similarity: cosine between a token and the same position's token at the first layer, compared
  with the mean cosine to the other positions of the same image.
"""

import torch
from torch.nn import functional as F


def collect(model, images, layers):
    """{layer: [batch, patches, dim]} normalised block inputs (what attention reads), prefix tokens removed."""
    store, handles = {}, []
    prefix = model.num_prefix_tokens
    for layer in layers:
        norm = model.blocks[layer - 1].norm1
        handles.append(norm.register_forward_hook(
            lambda module, inputs, output, layer=layer: store.__setitem__(layer, output[:, prefix:].detach())))
    try:
        with torch.no_grad():
            model(images)
    finally:
        for handle in handles:
            handle.remove()
    return store


def self_similarity(tokens, first):
    """Per image: mean cosine to the same position at the first layer, and mean cosine to other positions."""
    a, b = F.normalize(tokens.float(), dim=-1), F.normalize(first.float(), dim=-1)
    cosine = a @ b.transpose(1, 2)  # [batch, patches, patches]
    same = cosine.diagonal(dim1=1, dim2=2).mean(1)
    n = cosine.shape[1]
    other = (cosine.sum((1, 2)) - cosine.diagonal(dim1=1, dim2=2).sum(1)) / (n * (n - 1))
    return same, other


def train_position_probe(train, test, epochs=30, batch=4096, lr=1e-2, seed=0):
    """train/test: [images, patches, dim]. Returns (test accuracy, mean grid distance of the errors).

    A linear softmax classifier on standardised features; positions are the patch indices.
    """
    generator = torch.Generator(device='cpu').manual_seed(seed)
    device = train.device
    patches, dim = train.shape[1], train.shape[2]
    x = train.reshape(-1, dim).float()
    mean, std = x.mean(0), x.std(0).clamp_min(1e-6)
    x = (x - mean) / std
    y = torch.arange(patches, device=device).repeat(train.shape[0])
    linear = torch.nn.Linear(dim, patches).to(device)
    torch.manual_seed(seed)
    linear.reset_parameters()
    optimizer = torch.optim.Adam(linear.parameters(), lr=lr)
    for _ in range(epochs):
        for chunk in torch.randperm(len(x), generator=generator).to(device).split(batch):
            optimizer.zero_grad()
            F.cross_entropy(linear(x[chunk]), y[chunk]).backward()
            optimizer.step()
    with torch.no_grad():
        xt = (test.reshape(-1, dim).float() - mean) / std
        yt = torch.arange(patches, device=device).repeat(test.shape[0])
        predicted = linear(xt).argmax(-1)
    grid = int(patches ** 0.5)
    rows, cols = torch.div(predicted, grid, rounding_mode='floor'), predicted % grid
    true_rows, true_cols = torch.div(yt, grid, rounding_mode='floor'), yt % grid
    distance = torch.hypot((rows - true_rows).float(), (cols - true_cols).float())
    return float((predicted == yt).float().mean()), float(distance.mean())
