from mita_probe.runtime import configure_runtime

configure_runtime()

import pytest
import torch


@pytest.fixture(autouse=True)
def reproducible_tensors():
    torch.set_num_threads(2)
    torch.manual_seed(0)
