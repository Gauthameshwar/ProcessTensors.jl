#!/usr/bin/env bash
# Serial, single-thread C++ vs Julia vs exact-diagonalization accuracy.
# Small doable models use the published Table S.3.1 discretisation
# (central spin: dt=0.01, eps=1e-10; spin-boson: dt=0.1, C=0.1, eps=1e-7)
# at N small enough for full Hilbert-space ED.
#
#   bash benchmark/JuliaVsC++/run_sanity_cpp_julia_ed.sh
#
# Optional: SANITY_MODELS="central_spin spinboson" SANITY_CPU=0

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACE_BIN="${ACE_BIN:-$ROOT/ACE/bin/ACE}"
SANITY_CPU="${SANITY_CPU:-0}"
MODELS="${SANITY_MODELS:-central_spin spinboson}"

# shellcheck source=cpp/bench_common.sh
source "$HERE/cpp/bench_common.sh"

if [[ ! -x "$ACE_BIN" ]]; then
  echo "ACE binary not found at $ACE_BIN" >&2
  echo "Build it first: cd ACE && make" >&2
  exit 1
fi

single_thread_env
print_ace_provenance
cat <<'EOF'
SANITY: C++ vs Julia vs ED
  Central spin: N=3, dt=0.01, te=1, eps=1e-10, fully polarised.
  Spin-boson:   N_E=2, M=5, dt=0.1, te=2, C=0.1, kBT=0.5, eps=1e-7.
  Julia compression defaults to :zipup_cpp. Both codes propagate so the
  reduced density matrices can be compared to exact exp(-i t H).

EOF

echo "Pinning Julia and ACE to CPU $SANITY_CPU with OMP/MKL/OpenBLAS=1"
echo

run_one() {
  local name="$1" script="$2"
  echo "=== $name ==="
  taskset -c "$SANITY_CPU" \
    env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        MKL_DYNAMIC=FALSE OMP_DYNAMIC=FALSE \
        ACE_BIN="$ACE_BIN" \
        JULIA_VS_CPP_SKIP_INSTANTIATE=1 \
        JULIA_VS_CPP_BLAS_THREADS=1 \
    julia -t 1 --project="$ROOT" "$script"
  echo
}

if [[ " $MODELS " == *" central_spin "* ]]; then
  run_one "central spin" "$HERE/sanity_central_spin/run_central_spin_sanity.jl"
fi
if [[ " $MODELS " == *" spinboson "* ]] || [[ " $MODELS " == *" spin_boson "* ]]; then
  run_one "spin-boson" "$HERE/sanity_spin_boson/run_spinboson_sanity.jl"
fi

echo "Accuracy summaries:"
[[ -f "$HERE/sanity_central_spin/results/summary.txt" ]] &&
  echo "  $HERE/sanity_central_spin/results/summary.txt"
[[ -f "$HERE/sanity_spin_boson/results/summary.txt" ]] &&
  echo "  $HERE/sanity_spin_boson/results/summary.txt"
