#!/usr/bin/env bash
# Parallel Julia ACE builds with LAPACK gesvd (ITensors alg="qr_iteration")
# for the partially polarised central spin and the kBT/Omega=0.5 spin boson,
# over the same N grids as the gesdd campaign.
#
# After the builds, write results/gesvd_vs_gesdd.csv with
#   cpp_gesvd_build_s, julia_gesvd_build_s, julia_gesdd_build_s
# for each construction. C++ and Julia gesdd times are read from the
# existing campaign CSVs; this script only remeasures Julia gesvd.
#
#   bash benchmark/JuliaVsC++/run_gesvd_comparison.sh
#
# Optional:
#   ACE_CPUS="0 1 2 3" NS_CS="5" NS_SB="5" \
#     bash benchmark/JuliaVsC++/run_gesvd_comparison.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HERE="$ROOT/benchmark/JuliaVsC++"
OUT="$HERE/results"
CASE_DIR="$OUT/julia/.gesvd_cases"
CSV="$OUT/gesvd_vs_gesdd.csv"
NS_CS="${NS_CS:-5 10 25 50 100}"
NS_SB="${NS_SB:-5 10 20 50 100}"
JULIA_BIN="${JULIA_BIN:-julia +1.12.7}"

# shellcheck source=cpp/bench_common.sh
source "$HERE/cpp/bench_common.sh"
single_thread_env

mkdir -p "$OUT" "$CASE_DIR"

case_done() {
  local fragment="$1"
  [[ -s "$fragment" ]] || return 1
  awk -F, 'NR == 2 && $1 != "" && $3 + 0 > 0 { found = 1 } END { exit !found }' "$fragment"
}

merge_fragment() {
  local fragment="$1"
  [[ -s "$fragment" ]] || return 0
  echo "Recorded $(basename "$fragment")"
}

write_comparison_csv() {
  python3 - "$CSV" "$CASE_DIR" \
    "$OUT/cpp/central_spin.csv" "$OUT/julia/central_spin.csv" \
    "$OUT/cpp/spinboson.csv" "$OUT/julia/spinboson.csv" << 'PY'
import csv
import sys
from pathlib import Path

out, case_dir, cpp_cs, julia_cs, cpp_sb, julia_sb = sys.argv[1:]

def rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))

def last_match(table, pred, what):
    found = None
    for row in table:
        if pred(row):
            found = row
    if found is None:
        raise SystemExit(f"missing reference row: {what}")
    return found

def num(row, *names):
    for name in names:
        if name in row and row[name] != "":
            return row[name]
    raise SystemExit(f"missing numeric column in {row}")

def near(text, target):
    return abs(float(text) - target) <= 1e-8 * max(1.0, abs(target))

cpp_cs_rows = rows(cpp_cs)
julia_cs_rows = rows(julia_cs)
cpp_sb_rows = rows(cpp_sb)
julia_sb_rows = rows(julia_sb)

header = [
    "model",
    "case",
    "N",
    "cpp_gesvd_build_s",
    "julia_gesvd_build_s",
    "julia_gesdd_build_s",
    "cpp_gesvd_maxdim",
    "julia_gesvd_maxdim",
    "julia_gesdd_maxdim",
]
written = []
fragments = sorted(Path(case_dir).glob("*.csv"))
if not fragments:
    raise SystemExit(f"no Julia gesvd fragments in {case_dir}")

for fragment in fragments:
    with fragment.open(newline="") as handle:
        sample = next(csv.DictReader(handle))
    model = sample["model"]
    n_modes = int(sample["N"])
    if model == "central_spin":
        cpp = last_match(
            cpp_cs_rows,
            lambda row: row["model"] == "central_spin"
            and row["polarisation"] == "partial"
            and int(row["N"]) == n_modes,
            f"cpp central_spin partial N={n_modes}",
        )
        old = last_match(
            julia_cs_rows,
            lambda row: row["model"] == "central_spin"
            and row["polarisation"] == "partial"
            and int(row["N"]) == n_modes
            and row["compression"] == "zipup_cpp",
            f"julia gesdd central_spin partial N={n_modes}",
        )
        case = "partial"
        n_key = "N"
    elif model == "spinboson":
        cpp = last_match(
            cpp_sb_rows,
            lambda row: row["model"] == "spinboson"
            and near(row["kBT_over_Omega"], 0.5)
            and int(row["N_modes"]) == n_modes,
            f"cpp spinboson kBT/Omega=0.5 N={n_modes}",
        )
        old = last_match(
            julia_sb_rows,
            lambda row: row["model"] == "spinboson"
            and near(row["kBT_over_Omega"], 0.5)
            and int(row["N_modes"]) == n_modes
            and row["compression"] == "zipup_cpp",
            f"julia gesdd spinboson kBT/Omega=0.5 N={n_modes}",
        )
        case = "0.5"
        n_key = "N_modes"
    else:
        raise SystemExit(f"unknown model {model} in {fragment}")
    written.append([
        model,
        case,
        n_modes,
        num(cpp, "build_s"),
        sample["julia_gesvd_build_s"],
        num(old, "build_s"),
        num(cpp, "maxdim", "max_bond_dimension"),
        sample["julia_gesvd_maxdim"],
        num(old, "maxdim", "max_bond_dimension"),
    ])

written.sort(key=lambda row: (row[0], int(row[2])))
with open(out, "w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(header)
    writer.writerows(written)
print(f"Wrote {out} ({len(written)} rows)")
PY
}

queue_launch() {
  local spec="$1" cpu="$2" model n_modes stem
  IFS=: read -r model n_modes <<< "$spec"
  stem="${model}_N${n_modes}"
  q_log="$CASE_DIR/${stem}.log"
  q_frag="$CASE_DIR/${stem}.csv"
  if [[ "$model" == "central_spin" ]]; then
    q_label="central-spin partial N=$n_modes"
  else
    q_label="spin-boson kBT/Omega=0.5 N=$n_modes"
  fi
  echo "CPU $cpu  <-  $q_label  (julia -t 1, BLAS=1, gesvd)"
  stream_prefixed "$q_log" "$q_label" \
    taskset -c "$cpu" env \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_DYNAMIC=FALSE \
    OMP_DYNAMIC=FALSE \
    SKIP_INSTANTIATE=1 \
    JULIA_VS_CPP_PRINT_PROVENANCE=0 \
    JULIA_VS_CPP_PIN_CPU="$cpu" \
    GESVD_MODEL="$model" \
    GESVD_N="$n_modes" \
    GESVD_FRAGMENT="$q_frag" \
    stdbuf -oL -eL $JULIA_BIN -t 1 --project="$ROOT/benchmark" "$HERE/run_gesvd_case.jl"
}

echo "Instantiating the Julia 1.12.7 benchmark environment"
# shellcheck disable=SC2086
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  $JULIA_BIN -t 1 --project="$ROOT/benchmark" -e '
    include(joinpath(ARGS[1], "common.jl"))
    pin_compute_threads!()
    method = ProcessTensors.ACE(svd_alg="qr_iteration")
    method.svd_alg == "qr_iteration" || error("ACE svd_alg was not stored")
    println("ACE svd_alg = ", method.svd_alg, " (LAPACK gesvd)")
    print_julia_provenance()
  ' "$HERE"

cases=()
for n_modes in $NS_CS; do
  frag="$CASE_DIR/central_spin_N${n_modes}.csv"
  if case_done "$frag"; then
    echo "Skipping completed Julia gesvd case central-spin partial N=$n_modes"
  else
    rm -f "$frag"
    cases+=("central_spin:$n_modes")
  fi
done
for n_modes in $NS_SB; do
  frag="$CASE_DIR/spinboson_N${n_modes}.csv"
  if case_done "$frag"; then
    echo "Skipping completed Julia gesvd case spin-boson kBT/Omega=0.5 N=$n_modes"
  else
    rm -f "$frag"
    cases+=("spinboson:$n_modes")
  fi
done

status=0
if (( ${#cases[@]} > 0 )); then
  need_cpu_list
  run_in_cpu_batches || status=$?
else
  echo "All requested Julia gesvd cases are already present in $CASE_DIR"
fi

write_comparison_csv || status=$?
exit "$status"
