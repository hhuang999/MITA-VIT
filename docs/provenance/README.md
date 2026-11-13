# Provenance of the original observations

The published `outputs/` are preserved observations from the original frozen-model study. No new inference run is substituted for them. The repository also includes the configurations, fixed samples, original protocol and historical run records needed to interpret the results.

## Main study records

[`runs.json`](runs.json) maps the 42 model/dataset combinations of the original study to their active config, authentic source directory, execution revision and recovered records. Its 232 records, including register-search records, are byte-for-byte copies from `study/v5` commit `1d66c0f7b39119e3ee7777ad7d086df68769d427`. Original names and settings describe those executions. Some fields remain unknown rather than being reconstructed after the fact, including launch command, driver version, complete environment freeze and execution-time checkpoint digest. Multiple sources are listed when probe-error fixes and mean predictions came from different runs.

[`historical/import_study_outputs.py`](historical/import_study_outputs.py) is the original import program, preserved as evidence of how historical study files became the active `outputs/` layout. It records renaming of matched controls, selection of layer-specific readings, and aggregation of value-norm sums. Numeric observations were not newly inferred. The `k_sweep` configuration was once named `fig2a_k_sweep`; the data and sampled images did not change.

[`artifacts.json`](artifacts.json) holds SHA-256 hashes of original observations, summary CSVs and fixed splits, plus CSV schemas, row counts and identifier-set digests. It also records semantic hashes of experiment configurations and abstract-syntax fingerprints of the core scientific code. CSV content hashes use decompressed bytes; the byte hash additionally distinguishes the exact gzip representation. The hashes were captured from the preserved observations, not learned from a new run.

The parameter-randomisation check in the appendix comes from the earlier study's `archive/outputs/sanity/` files at commit `7477ad5da9ab4d46f646ba2c0d2c3fe0316d9722`. The six copied run files now live under `outputs/sanity/`; `configs/sanity.yaml` and `src/fastweight_vit/experiments/sanity.py` preserve the original protocol and implementation. `results/sanity.csv` has SHA-256 `87d72799e193c66b9d2911d0dba451496ccf38a6fcc4c2191efc4c333f1a4839`, identical to the original summary. These observations are checked by `validate_research.py` and rebuilt by `make_tables.py`. Their original imported `checks.json` markers do not stand in for complete execution metadata.

## What is and is not regenerated

| Material | Location | Verification |
| --- | --- | --- |
| Original per-image observations, probes and arrays | `outputs/` | Immutable hashes, schemas, image identifiers and original records |
| Original main-study run records | `docs/provenance/records/` | Source revision and exact file hash |
| Fixed image and gallery selections | `data/splits/` | Exact file hashes and sample rules |
| Generated numerical summaries | `results/` except `results/register/` | Rebuild from saved observations and compare at absolute tolerance `1e-12` |
| Historical register pilot and formal summaries | `results/register/` | Exact CSV hashes and source links in its README |
| Figure and table displays | `paper/`, `scripts/display/` | Rebuild from the verified numerical summaries and saved gallery arrays |

The historical full-validation register accuracy and reduction, and the earlier Small/Tiny top-k contrast, use earlier protocols. [The register evidence guide](../../results/register/README.md) links their original program and records. [The registration record](../preregistration.md) distinguishes registered outcomes from analyses designed after seeing the main results.

Historical GPU verification files in [`historical/`](historical/) refer to code revisions `9bbee6f` and `0d47c04` and the NVIDIA L4 runs available at that time. They are retained as historical evidence; statements about paths or pending publication in those records describe their original context.
