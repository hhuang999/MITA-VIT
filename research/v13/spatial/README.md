# Frozen DeiT-S spatial readout: independent Oxford pet trimaps

This is a **new exploratory** analysis after the v12 registration. The question,
sample rule, controls, and primary metric were written in [PROTOCOL.md](PROTOCOL.md)
before the scores were run. The model was not trained or altered. The analysis
uses Oxford-IIIT Pet test trimaps, so its numbers must not be pooled with the
registered ImageNet-S or ClickMe evaluations.

## Materials and reproducibility

- Model: `timm/deit_small_patch16_224.fb_in1k`, pinned Hugging Face revision
  `91327a9c99f98fe6b524cd4d397b7226b80e1365`, via the repository's
  `load_deit('cuda')`. The downloaded `model.safetensors` SHA-256 was
  `1e747b4a8d0df2cfbd3c450e8c97685d867448ab0c2ddbfb34b6885f5cb23e5b`. The model
  has 22,050,664 parameters. [Pinned model source](https://huggingface.co/timm/deit_small_patch16_224.fb_in1k/tree/91327a9c99f98fe6b524cd4d397b7226b80e1365).
- Data: [Oxford-IIIT Pet, University of Oxford VGG](https://www.robots.ox.ac.uk/~vgg/data/pets/),
  `images.tar.gz` and `annotations.tar.gz` from the official site. Observed
  archive SHA-256 values were
  `67195c5e1c01f1ab5f9b6a5d22b8c27a580d896ece458917e61d459337fa318d`
  and
  `52425fb6de5c424942b7626b428656fcbd798db970a937df61750c0f1d358e91`,
  respectively. Raw archives and extracted files are in the ignored
  `build/paper-v13/spatial/` directory.
- Fixed sample: SHA-256 order of `v13-spatial-2026:` plus each name in
  `annotations/test.txt`; first 384. The exact names are in
  [selected_names.txt](selected_names.txt). One selected image,
  `wheaten_terrier_39`, has 196 object patches and zero background patches
  after the model crop, leaving 383 evaluable images.
- Runtime: Windows, `D:/anaconda/envs/vb/python.exe`, PyTorch 2.6.0+cu124,
  timm 1.0.30, torchvision 0.21.0+cpu, RTX 4060. The existing project
  dependency overlay `build/paper-v9/python-deps` supplied timm and compatible
  torchvision. Set `PYTHONPATH=build/paper-v9/python-deps;src` and
  `HF_HUB_CACHE=build/paper-v13/spatial/hf-hub` (absolute path) before running.
- Input preprocessing: bicubic antialiased resize of the shorter side to 248,
  center crop 224×224, ImageNet mean/std normalization. The trimap uses
  nearest-neighbor resize 248 and the identical crop. The repository's
  `token_foreground` defines valid/object 16×16 patches.

From the repository root, the following PowerShell commands stage the data and
reproduce the scores. Check the two archive hashes above after downloading.

```powershell
New-Item -ItemType Directory -Force build/paper-v13/spatial | Out-Null
curl.exe -L --fail -o build/paper-v13/spatial/images.tar.gz https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
curl.exe -L --fail -o build/paper-v13/spatial/annotations.tar.gz https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
tar -xzf build/paper-v13/spatial/images.tar.gz -C build/paper-v13/spatial
tar -xzf build/paper-v13/spatial/annotations.tar.gz -C build/paper-v13/spatial
$env:PYTHONPATH = 'build/paper-v9/python-deps;src'
$env:HF_HUB_CACHE = (Resolve-Path build/paper-v13/spatial).Path + '\hf-hub'
& 'D:/anaconda/envs/vb/python.exe' research/v13/spatial/run_spatial.py
& 'D:/anaconda/envs/vb/python.exe' research/v13/spatial/audit_results.py
```

`run_spatial.py` writes [per_image.csv](per_image.csv) and
[summary.json](summary.json). The later reciprocal-pair sensitivity analysis is
`python research/v13/spatial/audit_results.py`, writing
[pair_block_sensitivity.json](pair_block_sensitivity.json).

## Primary result

The number is mean same-distance-ring object enrichment over 383 paired images.
Intervals below are the prespecified image-level bootstrap 95% intervals for
each paired difference versus C3, except C3 versus R0.

| Reading | Mean enrichment | Paired difference | 95% interval |
| --- | ---: | ---: | ---: |
| R0 | 0.14064 | reference | — |
| C3 | 0.15964 | +0.01900 vs R0 | [+0.01550, +0.02273] |
| C3v (unprojected value norm) | 0.16346 | +0.00383 vs C3 | [+0.00263, +0.00512] |
| Norm only (per-head attention mass retained) | 0.11969 | −0.03994 vs C3 | [−0.04811, −0.03203] |
| Donor norm ranks, recipient values | 0.13731 | −0.02233 vs C3 | [−0.02614, −0.01869] |
| Same-ring norm shuffle | 0.13833 | −0.02131 vs C3 | [−0.02484, −0.01787] |
| Cross-head net output magnitude | 0.15632 | −0.00332 vs C3 | [−0.00480, −0.00193] |

The donor-rank control uses reciprocal pairs, so the image-level bootstrap
does not capture that shared control dependence. A **post-result sensitivity
analysis** resampled the 192 donor pairs as blocks and weighted by the number
of included images; C3 minus donor was +0.02233 with pair-block 95% interval
[+0.01867, +0.02614]. All listed contrast directions were stable under that
block bootstrap; its complete values are in the JSON.

For context, object/background AUROC was 0.7881 for R0, 0.8166 for C3, and
0.8222 for C3v. Mean valid object area was 0.507. The C3–R0 enrichment gain
was +0.0116 in 113 cat images and +0.0221 in 270 dog images (exploratory
subgroups). These results support **image-specific norm/attention spatial
alignment** as useful for this object readout. They do not isolate the output
projection as the reason: C3v did slightly better here. Accounting for
cross-head cancellation also did not improve the chosen object metric. The
study evaluates association with a foreground mask, not causal contribution
to the final classification.

## Implementation checks

For the first batch, the custom capture's C3 scores matched the repository's
`capture(..., per_head=True)` formula and model logits matched the reference
forward within numerical tolerance. The per-query per-destination net-output
norm satisfied the triangle-inequality upper bound from per-head magnitudes
throughout the run (maximum observed positive violation 0). All seven maps
used the same images, masks, and layers 9–12.
