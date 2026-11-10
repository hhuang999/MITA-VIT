# Experiments outside the paper

These experiments and earlier materials are not used by the final course paper.
Paths below keep their original layout. Shared build scripts are snapshots from before the cleanup.
The archived code is not meant to run from this folder.

To run the old experiments, check out commit
`b3ae06df55b264e3e32b40c9c132923fa117601c` in a separate clone.

| Item | Purpose |
| --- | --- |
| `single_layer` | Restrict one attention layer at a time |
| `part_reliance` and the PartImageNet loader | Compare head and body patches |
| `map_similarity` | Compare restricted and full maps |
| `order` | Compare another order of the corrections |
| `sanity` | Randomise model weights |
| `d2` | Discount scores without relocation |
| Other result tables and figures | Analyses and exports not used in the paper |
| `docs/` and old `paper/` guides | Earlier study rules, verification records and naming notes |
| `scripts/` and `reproduce/` | Original shared build and run scripts |
| `tests/` | Tests of archived functionality split from the active tests |

The paper uses the retained supplementary modules for contribution checks, MLP probes
and globality. Those shared modules stay active, including their additional analyses.
Local ignored writing materials were left on the author's machine.
