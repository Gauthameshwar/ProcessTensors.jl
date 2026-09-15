#!/usr/bin/env bash
# Sweep OpenBLAS/MKL threads for the ACE compressor figure panel (f).
# Do not taskset this launcher to a single core.
#
#   bash benchmark/ace_compressors/run_ace_thread_scaling.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
THREADS="${ACE_THREAD_GRID:-1 2 4 8 16 32}"
SAMPLES="${ACE_BENCH_SAMPLES:-1}"

cd "$ROOT"
for n in $THREADS; do
  echo "=== ACE_BLAS_THREADS=$n ==="
  env \
    OMP_NUM_THREADS="$n" \
    MKL_NUM_THREADS="$n" \
    OPENBLAS_NUM_THREADS="$n" \
    MKL_DYNAMIC=FALSE \
    OMP_DYNAMIC=FALSE \
    ACE_BLAS_THREADS="$n" \
    ACE_BENCH_SAMPLES="$SAMPLES" \
    julia -t 1 --project=. benchmark/ace_compressors/ace_thread_scaling.jl
done
