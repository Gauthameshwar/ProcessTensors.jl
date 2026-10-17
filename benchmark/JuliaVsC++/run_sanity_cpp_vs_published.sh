#!/usr/bin/env bash
# Serial, single-thread C++ ACE construction times at the published
# Table S.3.1 discretisation. dont_propagate and use_symmetric_Trotter stay
# on so the measured time is PT construction, not the paper's complete example.
#
#   bash benchmark/JuliaVsC++/run_sanity_cpp_vs_published.sh
#
# Optional: SANITY_CPU=0 SANITY_CASES="cs_polarised_N10_1e-10"

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACE_BIN="${ACE_BIN:-$ROOT/ACE/bin/ACE}"
SANITY_CPU="${SANITY_CPU:-0}"

# shellcheck source=cpp/extract_maxdim.sh
source "$HERE/cpp/extract_maxdim.sh"
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
SANITY: C++ construction vs Table S.3.1
  This script uses the published dt, te, N, cutoff, and couplings.
  dont_propagate=true, so local times are construction-only.
  published_s is the paper's complete-example wall time (build+contract+I/O).
  A healthy -O3 ACE binary on this server should beat published_s, but that
  is not a like-for-like construction comparison.

EOF

# run_id|model|param|published_s|published_display
ALL_CASES=(
  "cs_polarised_N10_1e-10|central_spin|$HERE/sanity_central_spin/published/polarised_N10_1e-10.param|70|1 min 10 s"
  "cs_polarised_N100_1e-10|central_spin|$HERE/sanity_central_spin/published/polarised_N100_1e-10.param|249|4 min 09 s"
  "cs_unpolarised_N100_1e-10|central_spin|$HERE/sanity_central_spin/published/unpolarised_N100_1e-10.param|507|8 min 27 s"
  "cs_unpolarised_N100_1e-13|central_spin|$HERE/sanity_central_spin/published/unpolarised_N100_1e-13.param|2452|40 min 52 s"
  "sb_harmonic_M5|spinboson|$HERE/sanity_spin_boson/published/independent_boson_T0.5_N101_M5.param|1217|20 min 17 s"
)

requested="${SANITY_CASES:-}"
cases=()
for spec in "${ALL_CASES[@]}"; do
  IFS='|' read -r run_id _model _param _pub _disp <<< "$spec"
  if [[ -z "$requested" ]] || [[ " $requested " == *" $run_id "* ]]; then
    cases+=("$spec")
  fi
done
if (( ${#cases[@]} == 0 )); then
  echo "No sanity cases selected. SANITY_CASES='$requested'" >&2
  exit 1
fi

CS_CSV="$HERE/sanity_central_spin/results/cpp_vs_published.csv"
SB_CSV="$HERE/sanity_spin_boson/results/cpp_vs_published.csv"
HEADER="run_id,model,elapsed_sec,maxdim,published_s,published_display,published_over_local,timing_scope,pinned_cpu,logfile"
mkdir -p "$HERE/sanity_central_spin/results" "$HERE/sanity_spin_boson/results"
printf '%s\n' "$HEADER" > "$CS_CSV"
printf '%s\n' "$HEADER" > "$SB_CSV"

echo "Pinning every ACE process to CPU $SANITY_CPU with OMP/MKL/OpenBLAS=1"
echo

for spec in "${cases[@]}"; do
  IFS='|' read -r run_id model param published_s published_display <<< "$spec"
  if [[ "$model" == "central_spin" ]]; then
    outdir="$HERE/sanity_central_spin/results"
    csv="$CS_CSV"
  else
    outdir="$HERE/sanity_spin_boson/results"
    csv="$SB_CSV"
  fi
  log="$outdir/${run_id}.log"
  timef="$outdir/${run_id}.time"
  echo "=== $run_id ==="
  echo "  param              : $param"
  echo "  published complete : $published_display ($published_s s)"
  echo "  local scope        : construction only (dont_propagate)"

  /usr/bin/time -f '%e %M %W' -o "$timef" \
    taskset -c "$SANITY_CPU" \
    env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        MKL_DYNAMIC=FALSE OMP_DYNAMIC=FALSE \
    "$ACE_BIN" "$param" > "$log" 2>&1 || {
      echo "ACE failed; see $log" >&2
      tail -n 30 "$log" >&2
      exit 1
    }

  parse_gnu_time "$timef"
  maxdim=$(maxdim_from_log "$log")
  ratio=$(ratio_or_blank "$published_s" "$elapsed_sec")
  echo "  local construct_s  : $elapsed_sec"
  echo "  D_max              : $maxdim"
  echo "  published/local    : ${ratio:-n/a}"
  echo "  peak_rss_kb        : $peak_rss_kb"
  echo
  echo "$run_id,$model,$elapsed_sec,$maxdim,$published_s,$published_display,$ratio,construction_only,$SANITY_CPU,$log" >> "$csv"
done

echo "wrote $CS_CSV"
echo "wrote $SB_CSV"
echo
echo "Reminder: published_s includes system propagation. Local elapsed_sec does not."
