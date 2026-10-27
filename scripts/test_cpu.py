"""Run local tests only after verifying CUDA is unavailable in this process."""
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import torch
import pytest

if torch.cuda.is_available():
    raise RuntimeError('CPU-only test guard failed; refusing to run tests')
raise SystemExit(pytest.main(sys.argv[1:]))
