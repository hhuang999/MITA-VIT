#!/usr/bin/env bash
# Run every v3 analysis and figure (decisions B09-B16) on downloaded results; CPU only.
# Usage: bash scripts/analyze_v3.sh [experiments/2026-10-29_v3_]
# Missing runs are skipped, so this can be re-run as results arrive.
set -uo pipefail
P=${1:-experiments/2026-10-29_v3_}
PY=${PY:-python}
ran() { [ -f "$1/checks.json" ]; }
for run in "$P"e4_*; do ran "$run" && $PY scripts/analyze_explanations.py "$run" > "$run/analysis.txt"; done
for run in "$P"e7_clickme_*; do ran "$run" && $PY scripts/analyze_human_alignment.py "$run" > "$run/analysis.txt"; done
for run in "$P"e8_parts_*; do ran "$run" && $PY scripts/analyze_parts.py "$run" > "$run/analysis.txt"; done
for run in "$P"sanity_*; do ran "$run" && $PY scripts/analyze_sanity.py "$run" > "$run/analysis.txt"; done
for run in "$P"e5_shifted_*; do ran "$run" && $PY scripts/analyze_shifted_crops.py "$run" > "$run/analysis.txt"; done
for run in "$P"e6_deletion_*; do ran "$run" && $PY scripts/analyze_matched_deletion.py --run "$run" > "$run/analysis.txt"; done
identity=$(for m in deit dino deit_b; do ran "${P}e3_identity_$m" && echo "${P}e3_identity_$m"; done)
[ -n "$identity" ] && $PY scripts/analyze_identity.py $identity --output "${P}e3_identity.csv"

mkdir -p figures
existing() { for run in "$@"; do [ -f "$run/analysis_comparisons.csv" ] && echo "$run"; done; }
runs=$(existing "${P}e4_imagenet_s_deit" "${P}e4_imagenet_s_dino" "${P}e4_imagenet_s_deit_b")
[ -n "$runs" ] && $PY scripts/plot_explanations.py $runs --output figures/figure8_explanations_imagenet_s
for d in hard_imagenet pet imagenet; do
  runs=$(existing "${P}e4_${d}_deit" "${P}e4_${d}_dino" "${P}e4_${d}_deit_b")
  [ -n "$runs" ] && $PY scripts/plot_explanations.py $runs --output "figures/figureS_explanations_$d"
done
runs=$(existing "${P}e7_clickme_deit" "${P}e7_clickme_dino" "${P}e7_clickme_deit_b")
[ -n "$runs" ] && $PY scripts/plot_human_parts_sanity.py human $runs --output figures/figure9_human
runs=$(existing "${P}e8_parts_deit" "${P}e8_parts_dino" "${P}e8_parts_deit_b")
[ -n "$runs" ] && $PY scripts/plot_human_parts_sanity.py parts $runs --output figures/figure10_parts
runs=$(for m in deit dino deit_b; do [ -f "${P}sanity_$m/analysis_sanity.csv" ] && echo "${P}sanity_$m"; done)
[ -n "$runs" ] && $PY scripts/plot_human_parts_sanity.py sanity $runs --output figures/figure11_sanity
shift_runs=$(for m in deit dino deit_b; do [ -f "${P}e5_shifted_$m/analysis_shift.csv" ] && echo "${P}e5_shifted_$m"; done)
deletion_runs=$(for d in imagenet_s hard_imagenet; do [ -f "${P}e6_deletion_$d/summary.csv" ] && echo "${P}e6_deletion_$d"; done)
if [ -n "$identity" ] && [ -n "$shift_runs" ] && [ -n "$deletion_runs" ]; then
  $PY scripts/plot_checks.py --identity $identity --shift $shift_runs --deletion $deletion_runs \
      --output figures/figure12_checks
fi
for m in deit dino deit_b; do
  g="${P}gallery_$m"
  [ -f "$g/gallery_imagenet_s.npz" ] || continue
  for d in imagenet_s clickme partimagenet hard_imagenet; do
    $PY scripts/plot_gallery.py methods "$g/gallery_$d.npz" --rows 6 --output "figures/figure13_gallery_${d}_$m"
  done
  $PY scripts/plot_gallery.py layers "$g/gallery_imagenet_s.npz" --rows 4 --output "figures/figure14_layers_$m"
done
echo ANALYSIS_DONE
