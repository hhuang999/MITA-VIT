# Candidate scientific figures

These are alternatives for author review. The v9 manuscript and PDF have not been changed.
Both candidates use existing observations and original uncertainty intervals and verdicts.

## Recommendation

Prioritise **Candidate 2**, which places each diagnostic directly over its correction and matched-control
outcomes. It replaces conceptual Table 1 plus diagnostic Figure 2. This makes the paper's diagnostic
contribution visible and keeps the mismatch between probe accuracy and relocation outcome explicit.

**Candidate 1** is a compact opening summary of contribution weighting: alignment gains, gains beyond
the matched control, and the deletion boundary. It replaces the existing opening figure. The tradeoff is
losing the opening network schematic; keep that schematic if the first page should emphasise the lens.
The two candidates intentionally reuse the C3 contrasts as an overview and as comparators in the
diagnostic framework; they are not independent replications.

The unchanged numerical main table is retained, including GAE, model-side variants, deletion and R_all.

## Files

Exports are in `../../output/figure_candidates/`:

- `01_repair_benefits_limits.pdf`, `.svg`, `.png`: Candidate 1, 139.7 x 47.0 mm.
- `02_diagnosis_correction_bridge.pdf`, `.svg`, `.png`: Candidate 2, 139.7 x 72.4 mm.
- `03_four_page_placement_preview.pdf`: isolated placement proof using both candidates.
- Selected source CSVs, original input hashes, per-figure alignment/collision/text reports and QA summary.

`figure_contract.md` records the scientific question, panel roles, data mapping, scope and exclusions.

## Rebuild and audit

Use the established Python 3.12 environment with Matplotlib, NumPy and PyMuPDF. From the repository root:

```powershell
$auditScripts = Join-Path $env:USERPROFILE '.codex/skills/nature-figure/scripts'
python paper/figure_candidates/build_figures.py --audit-scripts $auditScripts
python paper/figure_candidates/audit_candidates.py --audit-scripts $auditScripts
```

The placement helper uses an installed Tectonic executable and the same ignored NeurIPS style used for v9:

```powershell
python paper/figure_candidates/make_layout_preview.py --tectonic build/paper-v9/toolchain/tectonic.exe
```

The helper creates a separate manuscript copy under `build/figure-candidates/placement/`, moves the
displaced correction/control definitions into Method, updates figure/table references, and verifies that
the conclusion ends on page 4. Original body font sizes and page geometry are retained. The deliverable
preview contains the four main-content pages only, not references or appendices, and is not an adopted draft.

## Interpretation and QA

- Grey effect-size bands show the registered practical-margin range, not equivalence or an automatic null result.
- Filled/open effect markers preserve the original higher/lower versus comparable category, respectively.
- Candidate 2 circles compare against received attention; diamonds compare against the matched control.
- All uncertainty is the original 95% image-bootstrap interval, with 1,000 resamples. Images are the resampling
  units; intervals do not represent variation across independently trained model seeds.
- Enrichment n: 12,150 (DeiT-S/B), 12,144 (DINO); ClickMe n: 17,408; deletion n: 2,000.
- Supervised and self-supervised frozen checkpoints have one training realization each. No inference,
  training, probe fitting, resampling, new comparison or new significance decision is performed.
- No multiple-comparison correction is introduced. Diagnostic--benefit interpretation remains post hoc.
- All selected means, interval endpoints and verdicts match the existing rows, with no silent image sampling.
- Both figures pass rendered panel alignment at 1.5 pt and the 5 pt glyph-floor audit
  (actual minima: 7 pt in Candidate 1, 6.4 pt in Candidate 2)
  and collision QA (zero FAIL/WARN). Full-colour, monochrome and final-size manuscript views are inspected.
- Source preflight warnings about the 139.7 mm width and PNG/TIFF defaults are accepted: this is the actual
  NeurIPS column width and PDF/SVG are the authoritative vector assets; PNG is only a 300-dpi preview.

## Caption suggestions

**Candidate 1, main-text version.** Contribution weighting (C3): (a) gains over received attention;
(b) gains beyond zeroing high-norm tokens; (c) deletion boundary. Points/bars: original paired means/95%
intervals. Filled/open: higher/lower versus comparable under the registered rule. Grey: practical-margin
range (0.02; deletion 0.05).

**Candidate 2, main-text version.** Diagnostic proxies (a--c) and corresponding corrections (d--f), matched
by column. Points/bands/bars: original means and 95% intervals. Circle/diamond: versus received
attention/matched control. Filled/open: higher/lower versus comparable. Grey: layers 9--12 (top);
plus/minus 0.02 margin (bottom).
