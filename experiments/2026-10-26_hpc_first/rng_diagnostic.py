"""Check repeated seeded CUDA multinomial draws on the recorded hardware."""
import hashlib
import json
import sys
from pathlib import Path

import torch

torch.set_num_threads(2)
torch.use_deterministic_algorithms(True)
weights = torch.ones(32 * 6 * 16, 197, device="cuda")
digests = []
for _ in range(2):
    generator = torch.Generator(device="cuda").manual_seed(0)
    draws = [torch.multinomial(weights, 16, replacement=False, generator=generator)
             for _ in range(12)]
    values = torch.stack(draws).cpu().numpy()
    digests.append(hashlib.sha256(values.tobytes()).hexdigest())
report = {
    "gpu": torch.cuda.get_device_name(0),
    "multiprocessors": torch.cuda.get_device_properties(0).multi_processor_count,
    "torch": torch.__version__,
    "seed": 0,
    "weights_shape": list(weights.shape),
    "k": 16,
    "draw_calls": 12,
    "repeated_hashes": digests,
    "same_device_repeat_identical": digests[0] == digests[1],
}
Path(sys.argv[1]).write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
