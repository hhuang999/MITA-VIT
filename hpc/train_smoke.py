import time

import torch


torch.manual_seed(2026)
torch.set_num_threads(2)

x = torch.randn(4096, 32)
target = x @ torch.linspace(-1, 1, 32).unsqueeze(1) + 0.05 * torch.randn(4096, 1)
model = torch.nn.Linear(32, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

with torch.no_grad():
    initial_loss = torch.nn.functional.mse_loss(model(x), target).item()

start = time.monotonic()
steps = 0
while time.monotonic() - start < 60:
    optimizer.zero_grad()
    loss = torch.nn.functional.mse_loss(model(x), target)
    loss.backward()
    optimizer.step()
    steps += 1

with torch.no_grad():
    final_loss = torch.nn.functional.mse_loss(model(x), target).item()

elapsed = time.monotonic() - start
print(f"elapsed_seconds={elapsed:.1f}")
print(f"steps={steps}")
print(f"initial_mse={initial_loss:.6f}")
print(f"final_mse={final_loss:.6f}")
assert final_loss < initial_loss / 10
