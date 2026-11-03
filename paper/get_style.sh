#!/usr/bin/env bash
# Download the official NeurIPS 2025 style file next to main.tex (third-party; not kept in Git).
set -euo pipefail
cd "$(dirname "$0")"
tmp=$(mktemp -d)
curl -sSfL -o "$tmp/styles.zip" https://media.neurips.cc/Conferences/NeurIPS2025/Styles.zip
unzip -q -o "$tmp/styles.zip" -d "$tmp"
cp "$tmp/Styles/neurips_2025.sty" .
rm -r "$tmp"
echo "neurips_2025.sty ready; build with: latexmk -pdf main.tex"
