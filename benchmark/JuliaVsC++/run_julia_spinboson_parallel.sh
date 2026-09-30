#!/usr/bin/env bash
# One julia -t 1 / BLAS=1 / zipup_cpp process per (T, N_E) point, each pinned
# to an exclusive CPU from the available pool. Skip a CSV row only when
# N_E, cutoff, compressor, and thread counts all match.
#
#   bash benchmark/JuliaVsC++/run_julia_spinboson_parallel.sh
#
# Optional:
#   ACE_CPUS="16 17 18 19" TEMPS="0 0.5" NS="5 10" \
#     bash benchmark/JuliaVsC++/run_julia_spinboson_parallel.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HERE="$ROOT/benchmark/JuliaVsC++"
OUT="$HERE/results/julia"
CSV="$OUT/spinboson.csv"
CASE_DIR="$OUT/.spinboson_cases"
TEMPS="${TEMPS:-0 0.5 1.0 3.0}"
NS="${NS:-5 10 20 50 100}"
CUTOFF="${CUTOFF:-1e-8}"
COMPRESSION="${JULIA_VS_CPP_COMPRESSION:-zipup_cpp}"
THREADS=1

# shellcheck source=cpp/bench_common.sh
source "$HERE/cpp/bench_common.sh"
single_thread_env

mkdir -p "$OUT" "$CASE_DIR"

case_done() {
  local csv="$1" temperature="$2" n_modes="$3"
  [[ -s "$csv" ]] && awk -F, -v T="$temperature" -v N="$n_modes" \
    -v CUT="$CUTOFF" -v COMP="$COMPRESSION" -v TH="$THREADS" '
    function val(name) { return (name in idx && idx[name] <= NF) ? $(idx[name]) : "" }
    function feq(a, b) {
      da = a + 0; db = b + 0
      d = da > db ? da - db : db - da
      s = (da > db ? da : db); if (s < 1) s = 1
      return d / s < 1e-8
    }
    NR == 1 {
      for (i = 1; i <= NF; i++) idx[$i] = i
      next
    }
    val("model") != "spinboson" { next }
    !feq(val("kBT_over_Omega"), T) { next }
    val("N_modes") + 0 != N + 0 { next }
    !feq(val("threshold"), CUT) { next }
    tolower(val("compression")) != tolower(COMP) { next }
    {
      th = val("blas_threads")
      if (th == "") th = val("nthreads")
      if (th == "") th = val("omp_threads")
      if (th == "") next
      if (th + 0 == TH + 0) found = 1
    }
    END { exit !found }
  ' "$csv"
}

merge_fragment() {
  local fragment="$1"
  [[ -s "$fragment" ]] || return
  while IFS=, read -r code model temperature n_modes rest; do
    [[ "$code" == "code" || "$model" != "spinboson" ]] && continue
    if ! case_done "$CSV" "$temperature" "$n_modes"; then
      echo "$code,$model,$temperature,$n_modes,$rest" >> "$CSV"
      echo "Merged Julia case kBT/Omega=$temperature N_modes=$n_modes"
    fi
  done < "$fragment"
}

echo "Instantiating the shared Julia benchmark environment and upgrading CSV columns"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  julia -t 1 --project="$ROOT/benchmark" -e '
    include(joinpath(ARGS[1], "common.jl"))
    pin_compute_threads!()
    print_julia_provenance()
    print_timing_policy()
    mkpath(joinpath(ARGS[1], "results", "julia"))
    ensure_csv_header!(joinpath(ARGS[1], "results", "julia", "spinboson.csv"), SPINBOSON_CSV_HEADER)
  ' "$HERE"

shopt -s nullglob
for fragment in "$CASE_DIR"/*.csv; do
  merge_fragment "$fragment"
done

cases=()
echo "Skip requires N_E, cutoff=$CUTOFF, compression=$COMPRESSION, threads=$THREADS"
for temperature in $TEMPS; do
  for n_modes in $NS; do
    if case_done "$CSV" "$temperature" "$n_modes"; then
      echo "Skipping completed Julia case kBT/Omega=$temperature N_modes=$n_modes"
    else
      cases+=("$temperature:$n_modes")
    fi
  done
done

if (( ${#cases[@]} == 0 )); then
  echo "All requested Julia spin-boson cases are already present in $CSV"
  exit 0
fi

queue_launch() {
  local spec="$1" cpu="$2" temperature n_modes stem
  IFS=: read -r temperature n_modes <<< "$spec"
  stem="spinboson_T${temperature}_N${n_modes}"
  q_log="$CASE_DIR/${stem}.log"
  q_frag="$CASE_DIR/${stem}.csv"
  q_label="kBT/Omega=$temperature N_modes=$n_modes"
  rm -f "$q_frag"
  echo "CPU $cpu  <-  $q_label  (julia -t 1, BLAS=1, $COMPRESSION)"
  taskset -c "$cpu" env \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_DYNAMIC=FALSE \
    OMP_DYNAMIC=FALSE \
    SKIP_INSTANTIATE=1 \
    JULIA_VS_CPP_PRINT_PROVENANCE=0 \
    JULIA_VS_CPP_PIN_CPU="$cpu" \
    JULIA_VS_CPP_COMPRESSION="$COMPRESSION" \
    JULIA_VS_CPP_TEMPS="$temperature" \
    JULIA_VS_CPP_NS="$n_modes" \
    JULIA_VS_CPP_OUTPUT=".spinboson_cases/${stem}.csv" \
    julia -t 1 --project="$ROOT/benchmark" "$HERE/run_julia_spinboson.jl" \
    > "$q_log" 2>&1 &
  q_pid=$!
}

need_cpu_list
run_in_cpu_batches
echo "Wrote $CSV"
