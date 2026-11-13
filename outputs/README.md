# Immutable original outputs

These are the original observations imported from the frozen study, not new inference runs.
Their hashes, schemas, source revisions, model/dataset mappings, authentic records and import transformations are
documented in [provenance](../docs/provenance/README.md). Minimal imported `checks.json` markers alone certify nothing.

Run `python scripts/validate_research.py --strict` to check completeness and integrity.
`python scripts/rebuild_results.py --destination build/rebuilt-results` reads these files and writes to a fresh build folder.
New GPU runs go to `outputs/rerun/` or `outputs/smoke/`; the runner refuses original output folders.
Preserve batch sizes: the ClickMe other-image control pairs within each original batch.
