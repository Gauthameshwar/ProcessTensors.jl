#!/usr/bin/env bash
# Time C++ ACE process-tensor construction for the three central-spin
# polarisations and N in {5,10,25,50,100}.
#
# Physics parameters are unchanged. Each case is pinned to one exclusive CPU
# with OMP/MKL/OpenBLAS threads forced to 1.
#
# Run from the repository root:
#   bash benchmark/JuliaVsC++/run_cpp_central_spin.sh
#
# Optional:
#   ACE_CPUS="0 1 2 3" NS="5 10" POLS="polarised" \
#     bash benchmark/JuliaVsC++/run_cpp_central_spin.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ACE_BIN="${ACE_BIN:-$ROOT/ACE/bin/ACE}"
ANALYZE="${ANALYZE:-$ROOT/ACE/tools/PTB_analyze}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPP="$HERE/cpp"
OUT="$HERE/results/cpp"
CASE_DIR="$OUT/.central_spin_cases"
NS="${NS:-5 10 25 50 100}"
POLS="${POLS:-polarised partial unpolarised}"
DT=0.1
TE=20
CUTOFF=1e-10
NSTEPS=200
HEADER="code,model,polarisation,N,Jk,dt,te,threshold,elapsed_sec,maxdim,logfile,ptfile,setup_s,build_s,contract_s,io_s,total_s,peak_rss_kb,swap_io,timing_scope,omp_threads,mkl_threads,pinned_cpu,max_bond_dimension,final_trace_error,trajectory_checksum,max_reference_error,validation_status"

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
CSV="$OUT/central_spin.csv"
ensure_csv "$CSV" "$HEADER"

print_ace_provenance
print_timing_policy

param_for() {
  case "$1" in
    polarised)   printf '%s\n' "$CPP/central_spin_polarised.param" ;;
    partial)     printf '%s\n' "$CPP/central_spin_partial.param" ;;
    unpolarised) printf '%s\n' "$CPP/central_spin_unpolarised.param" ;;
    *) echo "unknown polarisation: $1" >&2; return 1 ;;
  esac
}

seed_for() {
  case "$1" in
    polarised) printf 'n/a (all +z)' ;;
    *) printf 'RandomSpin_seed 1' ;;
  esac
}

polarization_for() {
  case "$1" in
    polarised) printf 'inf (+z)' ;;
    partial) printf 'b=20' ;;
    unpolarised) printf 'b=0' ;;
  esac
}

case_done() {
  local csv="$1" pol="$2" n="$3"
  [[ -s "$csv" ]] && awk -F, -v pol="$pol" -v n="$n" '
    NR > 1 && $2 == "central_spin" && $3 == pol && $4 + 0 == n + 0 { found = 1 }
    END { exit !found }
  ' "$csv"
}

merge_fragment() {
  local fragment="$1"
  [[ -s "$fragment" ]] || return
  local pol n
  while IFS=, read -r code model pol n rest; do
    [[ "$code" == "code" || "$model" != "central_spin" ]] && continue
    if ! case_done "$CSV" "$pol" "$n"; then
      echo "$code,$model,$pol,$n,$rest" >> "$CSV"
      echo "Merged C++ case polarisation=$pol N=$n"
    fi
  done < "$fragment"
}

run_one_case() {
  local pol="$1" N="$2" cpu="$3"
  local Jk tag log timef pt fragment param
  Jk=$(awk -v N="$N" 'BEGIN { printf "%.10g\n", 1 / N }')
  tag="central_${pol}_N${N}"
  log="$OUT/${tag}.log"
  timef="$OUT/${tag}.time"
  pt="$OUT/${tag}.pt"
  fragment="$CASE_DIR/${tag}.csv"
  param=$(param_for "$pol")

  echo "=== C++ ACE  polarisation=$pol  N=$N  J_k=$Jk  cpu=$cpu ==="
  print_run_parameters \
    "cpp_${tag}" central_spin "$pol" "$N" 2 "$(polarization_for "$pol")" \
    "$DT" "$TE" "$NSTEPS" "$CUTOFF" true "$(seed_for "$pol")" n/a "J_k=$Jk" "$cpu"

  run_ace_pinned "$cpu" "$log" "$timef" \
    "$param" \
    -RandomSpin_N_modes "$N" \
    -RandomSpin_J_max "$Jk" \
    -RandomSpin_J_min "$Jk" \
    -dont_propagate true \
    -write_PT "$pt" \
    -RandomSpin_print_initial "$OUT/${tag}_orientations.txt"

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
  echo "cpp,central_spin,$pol,$N,$Jk,$DT,$TE,$CUTOFF,$elapsed_sec,$maxdim,$log,$pt,,$build_s,$contract_s,$io_s,$elapsed_sec,$peak_rss_kb,$swap_io,build+write_PT,${OMP_NUM_THREADS:-1},${MKL_NUM_THREADS:-1},$cpu,$maxdim,n/a,n/a,n/a,$status" >> "$fragment"
}

shopt -s nullglob
for fragment in "$CASE_DIR"/*.csv; do
  merge_fragment "$fragment"
done

cases=()
for pol in $POLS; do
  param_for "$pol" >/dev/null
  for N in $NS; do
    if case_done "$CSV" "$pol" "$N"; then
      echo "Skipping completed C++ case polarisation=$pol N=$N"
    else
      cases+=("$pol:$N")
    fi
  done
done

if (( ${#cases[@]} == 0 )); then
  echo "All requested C++ central-spin cases are already present in $CSV"
  exit 0
fi

queue_launch() {
  local spec="$1" cpu="$2" pol N stem
  IFS=: read -r pol N <<< "$spec"
  stem="central_${pol}_N${N}"
  q_log="$CASE_DIR/${stem}.run.log"
  q_frag="$CASE_DIR/${stem}.csv"
  q_label="polarisation=$pol N=$N"
  rm -f "$q_frag"
  echo "Launching C++ $q_label on exclusive CPU $cpu"
  ( run_one_case "$pol" "$N" "$cpu" ) > "$q_log" 2>&1 &
  q_pid=$!
}

need_cpu_list
run_in_cpu_batches
echo "wrote $CSV"
