#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
#
# Build "FlyDSL for Dummies" as a printable PDF using pandoc + a LaTeX engine.
#
# Requirements (pick one PDF engine):
#   - pandoc      (https://pandoc.org)
#   - xelatex (default) or pdflatex, via texlive
#       Debian/Ubuntu:  sudo apt-get install pandoc texlive-xetex texlive-latex-recommended
#   - OR typst, a single self-contained binary (no system LaTeX):
#       PDF_ENGINE=typst bash build_pdf.sh
#       A pip-only setup works too: `pip install pypandoc_binary typst` then put a
#       `typst` CLI on PATH (see docs). Uses metadata_typst.yaml for hex colors.
#   For a lighter setup you can instead render to HTML (see --html below) and
#   print-to-PDF from a browser.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

OUT="${1:-flydsl_for_dummies.pdf}"
ENGINE="${PDF_ENGINE:-xelatex}"

# Chapters, in order. Keep this list in sync with new files.
CHAPTERS=(
  00_preface.md
  01_mental_model.md
  02_compilation_pipeline.md
  03_control_flow.md
  04_types_and_values.md
  05_layout_algebra.md
  06_tiling_partitioning.md
  07_data_movement.md
  08_mma.md
  09_loads_stores_intrinsics.md
  10_mfma_intrinsics.md
  11_escape_hatches.md
  12_worked_examples.md
  13_debugging.md
  14_reference.md
)

if [[ "${1:-}" == "--html" ]]; then
  OUT="flydsl_for_dummies.html"
  echo "Rendering HTML -> $OUT"
  pandoc metadata.yaml "${CHAPTERS[@]}" \
    --standalone --toc --toc-depth=2 --number-sections \
    --highlight-style=tango \
    -o "$OUT"
  echo "Done. Open $OUT and print-to-PDF from your browser if you lack LaTeX."
  exit 0
fi

if ! command -v pandoc >/dev/null 2>&1; then
  echo "ERROR: pandoc not found. Install pandoc, or run: $0 --html" >&2
  exit 1
fi

# The typst engine is a single self-contained binary (no system LaTeX needed).
# It reads link colors as hex strings, so we layer metadata_typst.yaml on top of
# metadata.yaml to replace the LaTeX color names (RoyalBlue, ...) used by xelatex.
if [[ "$ENGINE" == "typst" ]]; then
  echo "Rendering PDF (typst) -> $OUT"
  pandoc metadata.yaml metadata_typst.yaml "${CHAPTERS[@]}" \
    --pdf-engine=typst \
    --toc --toc-depth=2 --number-sections \
    --syntax-highlighting=tango \
    -o "$OUT"
  echo "Done: $HERE/$OUT"
  exit 0
fi

echo "Rendering PDF ($ENGINE) -> $OUT"
pandoc metadata.yaml "${CHAPTERS[@]}" \
  --pdf-engine="$ENGINE" \
  --toc --toc-depth=2 --number-sections \
  --highlight-style=tango \
  -V colorlinks=true \
  -o "$OUT"

echo "Done: $HERE/$OUT"
