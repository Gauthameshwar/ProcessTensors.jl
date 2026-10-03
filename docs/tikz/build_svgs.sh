#!/bin/sh
# Compile each standalone TikZ figure to an SVG under docs/src/assets/theory.
# Inkscape reads the PDF. dvisvgm --pdf cannot open these files here.
set -eu
cd "$(dirname "$0")"
out="../src/assets/theory"
mkdir -p "$out"
for name in \
  pt_anatomy \
  quantum_channel \
  channel_to_multitime_process \
  column_major_vectorisation \
  mpo_to_liouville_mps \
  closing_a_leg \
  process_contractions \
  tensor_contractions \
  mps_vocabulary
do
  pdflatex -interaction=nonstopmode "${name}.tex" >"${name}.build.log"
  inkscape "${name}.pdf" --export-type=svg --export-filename="${out}/${name}.svg" \
    || test -s "${out}/${name}.svg"
done
