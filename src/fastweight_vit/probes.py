"""Position probes: does token j still stand for patch j at layer l? (premise M2; correction C2)

A probe reads the tokens each attention layer reads (the block input after norm1, prefix tokens removed) and
predicts the token's grid position (196 classes; chance 1/196). `collect` gathers those tokens; the linear probe
is the one C2 uses; the two-layer MLP probe is the capacity check of the exploratory analysis.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
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


@dataclass
class PositionProbe:
    """Linear 196-way position classifier on standardised tokens."""
    mean: torch.Tensor
    std: torch.Tensor
    weight: torch.Tensor  # [positions, dim]
    bias: torch.Tensor    # [positions]

    def logits(self, tokens):
        x = (tokens.float() - self.mean) / self.std
        return x @ self.weight.T + self.bias

    def probs(self, tokens):
        """[..., patches, dim] -> P(position | token) [..., patches, positions]."""
        return self.logits(tokens).softmax(-1)

    def to(self, device):
        return PositionProbe(*(t.to(device) for t in (self.mean, self.std, self.weight, self.bias)))

    def arrays(self, prefix):
        return {f'{prefix}_{name}': getattr(self, name).cpu().numpy()
                for name in ('mean', 'std', 'weight', 'bias')}

    @classmethod
    def from_arrays(cls, arrays, prefix):
        return cls(*(torch.from_numpy(np.asarray(arrays[f'{prefix}_{name}']))
                     for name in ('mean', 'std', 'weight', 'bias')))


def fit_position_probe(train, epochs=30, batch=4096, lr=1e-2, seed=0):
    """train: [images, patches, dim] -> PositionProbe (Adam, full-batch shuffles of `batch` tokens, seeded)."""
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
    return PositionProbe(mean.detach(), std.detach(), linear.weight.detach().clone(), linear.bias.detach().clone())


class MLPProbe(nn.Module):
    def __init__(self, dim, hidden=1024, positions=196):
        super().__init__()
        self.register_buffer('mean', torch.zeros(dim))
        self.register_buffer('std', torch.ones(dim))
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, positions))

    def logits(self, tokens):
        return self.net((tokens.float() - self.mean) / self.std)

    @torch.no_grad()
    def probs(self, tokens):
        return self.logits(tokens).softmax(-1)


def fit_mlp_probe(train, validation, hidden=1024, epochs=30, batch=4096, lr=1e-2, seed=0):
    """train / validation: [images, patches, dim]. Returns (probe at the best validation epoch, history)."""
    generator = torch.Generator(device='cpu').manual_seed(seed)
    device = train.device
    patches, dim = train.shape[1], train.shape[2]
    x = train.reshape(-1, dim).float()
    torch.manual_seed(seed)
    probe = MLPProbe(dim, hidden, patches).to(device)
    probe.mean.copy_(x.mean(0))
    probe.std.copy_(x.std(0).clamp_min(1e-6))
    y = torch.arange(patches, device=device).repeat(train.shape[0])
    target = torch.arange(patches, device=device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    best, best_state, history = -1., None, []
    for _ in range(epochs):
        probe.train()
        for chunk in torch.randperm(len(x), generator=generator).to(device).split(batch):
            optimizer.zero_grad()
            F.cross_entropy(probe.logits(x[chunk]), y[chunk]).backward()
            optimizer.step()
        probe.eval()
        accuracy = float((probe.probs(validation).argmax(-1) == target).float().mean())
        history.append(accuracy)
        if accuracy > best:
            best, best_state = accuracy, {key: value.clone() for key, value in probe.state_dict().items()}
    probe.load_state_dict(best_state)
    return probe.eval(), history
