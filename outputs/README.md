# Immutable original outputs

These are the original observations imported into the release layout, not new inference runs.
Their hashes, schemas, source revisions, model/dataset mappings, authentic records and import transformations are
documented in [provenance](../docs/provenance/README.md). Minimal imported `checks.json` markers alone certify nothing.

Run `python scripts/validate_release.py --strict` to check completeness and integrity.
`bash reproduce/from_outputs.sh` reads these files and writes only to fresh `build/rebuilt/`.
Optional future GPU runs go to `outputs/rerun/` or `outputs/smoke/`; the runner refuses original output folders.
Preserve batch sizes: the ClickMe other-image control pairs within each original batch.
