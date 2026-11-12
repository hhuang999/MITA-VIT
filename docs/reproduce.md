# Reproducing the registered study

The paper combines a frozen, registered study with later exploratory analyses. This page documents the registered study and its preserved inputs. The exploratory Oxford Pets, CLS, geometry, and endpoint studies have their own commands in [experiments/README.md](../experiments/README.md). Run commands from the repository root with the installed project environment.

## Two reproducibility paths

**From saved observations, on CPU.** The original run files and restored parameter-randomisation observations are under `outputs/`. They are immutable reference observations. These commands check their hashes, effective protocols, fixed image identifiers, historical run records, and registered outcomes, then rebuild 20 CSV summaries in a fresh directory:

```sh
python scripts/validate_research.py --strict
python scripts/rebuild_results.py --outputs outputs --destination build/rebuilt-results
```

The rebuild compares numeric CSV fields with `results/` at absolute tolerance `1e-12`, and requires identical labels and verdicts. It never writes into `outputs/` or `results/`. Use a new destination for another rebuild. The paper figure and table renderers consume the verified results; see the root README for the paper-item command map.

**From data and model weights.** Obtain ImageNet-1k validation images under its access terms, ImageNet-S919 validation masks, and ClickMe validation shards, then download the three frozen checkpoints and the MiTA checkpoint:

```sh
python scripts/prepare_data.py imagenet imagenet_s clickme
python scripts/prepare_data.py --check
python scripts/download_models.py
python scripts/run.py --config configs/diagnose_probe.yaml --model deit --output outputs/rerun
python scripts/run.py --config configs/diagnose_mean_maps.yaml --model deit --output outputs/rerun
python scripts/run.py --config configs/table1.yaml --model deit --dataset imagenet_s --output outputs/rerun
```

The first run trains the position probes; the second creates held-out mean maps; the third evaluates received and modified readings. Run the corresponding ClickMe cell with `--dataset clickme`, or change `--model` to `deit_b` or `dino`. Model-side rows use `configs/table1_register.yaml` (DeiT-S) and `configs/table1_mita.yaml` (MiTA). `reproduce/full.sh` lists the full study order. GPU runs need a committed source tree because each new run records its source commit. They write under `outputs/rerun/`, separate from the preserved observations. Model and data scripts check checkpoint and download hashes; exact ImageNet and dataset revisions are pinned in `src/fastweight_vit/data/` and `src/fastweight_vit/models/`.

For a small trial of one config, pass `--smoke 16`. Full historical reproduction has substantial data and compute costs; the supplied results are the original observations, not a claim that every checkpoint was retrained or every historical run was repeated on this branch.

## Protocol and evidence map

| Evidence in the paper | Config / saved observations | Rebuilt summary |
| --- | --- | --- |
| Top-k execution and attention removal | `k_sweep`, `attention_off`; `outputs/{k_sweep,attention_off}/` | `results/k_sweep.csv`, `results/k_sweep_tests.csv` |
| Three diagnostics and per-layer ClickMe values | `diagnose_probe`, `diagnose_mean_maps`, `high_norm`, `clickme_layers` | `results/diagnose.csv`, `results/probe_accuracy.csv`, `results/radial.csv`, `results/value_norm.csv` |
| Received maps, corrections, controls, deletion, last-layer recovery, fixed gallery | `table1`; `outputs/table1/` | `results/table1_long.csv`, `results/correct_tests.csv`, `results/negative_controls.csv`, `results/last_layer.csv`; `data/splits/gallery_selection.json` |
| Test-time register and MiTA rows | `table1_register`, `table1_mita`; their `outputs/` folders | `results/variants.csv` |
| Parameter-randomisation check | `sanity`; `outputs/sanity/{deit,deit_b,dino}/sanity.csv` | `results/sanity.csv` |
| Probe-relocation analyses | `c2_analysis`, `supplementary_heldout`, `supplementary_eval` | `results/c2_analysis/`, `results/supplementary/` |

Each saved model/dataset run of the main study is mapped to authentic execution records in [provenance/runs.json](provenance/runs.json). [provenance/artifacts.json](provenance/artifacts.json) holds original-output, result, fixed-split, config, and scientific-code fingerprints. The records and import history are described in [provenance/README.md](provenance/README.md). Parameter-randomisation observations were restored from the earlier study at the same CSV byte hashes; their source is documented there separately.

## Registered rules and interpretation

The original timeline, predictions P1–P7, margins, bootstrap rule, and historical outcome ledger are preserved in [preregistration.md](preregistration.md). P4a/P4b are the registered negative controls (R1), P3 is the combined-readout prediction (R2), and P6 covers the model-side rows (R3). Primary paired comparisons were the combined reading versus received attention and each individual correction versus its matched control. Individual correction-versus-received differences had no registered direction. Diagnostic-to-benefit associations and later mechanism analyses are post hoc.

The original decision rule uses one value per image, a paired image bootstrap with 1,000 resamples and seed 0, and unadjusted 95% percentile intervals. A difference is comparable when its interval contains zero or its absolute mean falls below the practical margin: 0.02 for enrichment and ClickMe, 0.05 for deletion. `scripts/make_tables.py` implements the rule. The ClickMe shuffled-human control pairs examples within each original batch; changing batch size changes that pairing and is not an exact reproduction.

Two older claims retain historical evidence with their original protocols. The 91.5% formal register reduction and unchanged 79.8% accuracy come from [results/register/register_deit.csv](../results/register/register_deit.csv); [results/register/README.md](../results/register/README.md) links the fixed historical implementation and records. The Small/Tiny top-k training contrast used an earlier protocol and is documented in the [original B32 report at its source commit](https://github.com/hhuang999/vit-fastweight-lens/blob/1d66c0f7b39119e3ee7777ad7d086df68769d427/docs/analysis/v5_report.md). Neither claim is reconstructed by the current map-evaluation runner.
