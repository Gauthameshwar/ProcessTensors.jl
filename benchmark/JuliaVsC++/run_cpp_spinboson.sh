#!/usr/bin/env bash
# Time C++ ACE process-tensor construction for the Lorentzian spin-boson bath
# over temperatures and mode-grid sizes. Existing CSV cases are skipped.
#
# Physics parameters are unchanged. Each case is pinned to one exclusive CPU
# with OMP/MKL/OpenBLAS threads forced to 1.
#
# Run from the repository root:
#   bash benchmark/JuliaVsC++/run_cpp_spinboson.sh
#
# Optional:
#   ACE_CPUS="0 1 2 3" TEMPS="0.5 1.0" NS="5 10" \
#     bash benchmark/JuliaVsC++/run_cpp_spinboson.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ACE_BIN="${ACE_BIN:-$ROOT/ACE/bin/ACE}"
ANALYZE="${ANALYZE:-$ROOT/ACE/tools/PTB_analyze}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPP="$HERE/cpp"
OUT="$HERE/results/cpp"
CASE_DIR="$OUT/.spinboson_cases"
TEMPS="${TEMPS:-0 0.5 1.0 3.0}"
NS="${NS:-5 10 20 50 100}"
DT=0.1
TE=8
CUTOFF=1e-8
NSTEPS=80
M=5
C=0.2
HEADER="code,model,kBT_over_Omega,N_modes,M,C_over_Omega2,dt,te,threshold,elapsed_sec,maxdim,logfile,ptfile,setup_s,build_s,contract_s,io_s,total_s,peak_rss_kb,swap_io,timing_scope,omp_threads,mkl_threads,pinned_cpu,max_bond_dimension,final_trace_error,trajectory_checksum,max_reference_error,validation_status"

# shellcheck source=cpp/extract_maxdim.sh
source "$CPP/extract_maxdim.sh"
# shellcheck source=cpp/bench_common.sh
source "$CPP/bench_common.sh"

if [[ ! -x "$ACE_BIN" ]]; then
  echo "ACE binary not found at $ACE_BIN" >&2
  echo "Build it first: cd ACE && make" >&2
  exit 1
fi

single_thread_env
mkdir -p "$OUT" "$CASE_DIR"
CSV="$OUT/spinboson.csv"
ensure_csv "$CSV" "$HEADER"

print_ace_provenance
print_timing_policy

case_done() {
  local csv="$1" temperature="$2" n_modes="$3"
  [[ -s "$csv" ]] && awk -F, -v T="$temperature" -v N="$n_modes" '
    NR > 1 && $2 == "spinboson" && $3 + 0 == T + 0 && $4 + 0 == N + 0 { found = 1 }
    END { exit !found }
  ' "$csv"
}

merge_fragment() {
  local fragment="$1"
  [[ -s "$fragment" ]] || return
  local temperature n_modes
  while IFS=, read -r code model temperature n_modes rest; do
    [[ "$code" == "code" || "$model" != "spinboson" ]] && continue
    if ! case_done "$CSV" "$temperature" "$n_modes"; then
      echo "$code,$model,$temperature,$n_modes,$rest" >> "$CSV"
      echo "Merged C++ case kBT/Omega=$temperature N_modes=$n_modes"
    fi
  done < "$fragment"
}

run_one_case() {
  local T="$1" N="$2" cpu="$3"
  local tag log timef pt fragment
  tag="spinboson_T${T}_N${N}"
  log="$OUT/${tag}.log"
  timef="$OUT/${tag}.time"
  pt="$OUT/${tag}.pt"
  fragment="$CASE_DIR/${tag}.csv"

  echo "=== C++ ACE  spin-boson  kBT/Omega=$T  N_modes=$N  cpu=$cpu ==="
  print_run_parameters \
    "cpp_${tag}" spinboson thermal "$N" "$M" n/a \
    "$DT" "$TE" "$NSTEPS" "$CUTOFF" true n/a "$T" \
    "Lorentzian C=$C gamma=0.1 omega_c=1 omega in [0,7.5]" "$cpu"

  run_ace_pinned "$cpu" "$log" "$timef" \
    "$CPP/spinboson.param" \
    -temperature_unitless "$T" \
    -Boson_N_modes "$N" \
    -dont_propagate true \
    -write_PT "$pt"

  local io_s="0"
  if [[ -x "$ANALYZE" ]]; then
    io_s=$(time_cmd "$log" "$ANALYZE" -read_PT "$pt" || echo "0")
  fi

  local maxdim build_s status
  maxdim=$(maxdim_from_log "$log")
  build_s=$(awk -v t="$elapsed_sec" -v c="$contract_s" 'BEGIN { printf "%.6f", (t + 0) - (c + 0) }')
  if [[ -n "$maxdim" ]]; then
    status=PASS
  else
    status=FAIL
  fi

  print_results_block \
    "$maxdim" "" "$build_s" "$contract_s" "$io_s" "$elapsed_sec" \
    "$peak_rss_kb" "$swap_io" \
    "ACE wall (dont_propagate; write_PT included in total/build)"
  print_validation "$maxdim" "n/a (dont_propagate)" "n/a (dont_propagate)" "n/a (dont_propagate)" "$status"

  printf '%s\n' "$HEADER" > "$fragment"
  echo "cpp,spinboson,$T,$N,$M,$C,$DT,$TE,$CUTOFF,$elapsed_sec,$maxdim,$log,$pt,,$build_s,$contract_s,$io_s,$elapsed_sec,$peak_rss_kb,$swap_io,build+write_PT,${OMP_NUM_THREADS:-1},${MKL_NUM_THREADS:-1},$cpu,$maxdim,n/a,n/a,n/a,$status" >> "$fragment"
}

shopt -s nullglob
for fragment in "$CASE_DIR"/*.csv; do
  merge_fragment "$fragment"
done

cases=()
for T in $TEMPS; do
  for N in $NS; do
    if case_done "$CSV" "$T" "$N"; then
      echo "Skipping completed C++ case kBT/Omega=$T N_modes=$N"
    else
      cases+=("$T:$N")
    fi
  done
done

if (( ${#cases[@]} == 0 )); then
  echo "All requested C++ spin-boson cases are already present in $CSV"
  exit 0
fi

queue_launch() {
  local spec="$1" cpu="$2" T N stem
  IFS=: read -r T N <<< "$spec"
  stem="spinboson_T${T}_N${N}"
  q_log="$CASE_DIR/${stem}.run.log"
  q_frag="$CASE_DIR/${stem}.csv"
  q_label="kBT/Omega=$T N_modes=$N"
  rm -f "$q_frag"
  echo "Launching C++ $q_label on exclusive CPU $cpu"
  ( run_one_case "$T" "$N" "$cpu" ) > "$q_log" 2>&1 &
  q_pid=$!
}

need_cpu_list
run_in_cpu_batches
echo "wrote $CSV"
