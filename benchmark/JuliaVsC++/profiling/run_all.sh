#!/usr/bin/env bash
# Phase 1 ACE SVD audit. Partial central spin, N=5.
# One measured process at a time, pinned to an idle core whose sibling is idle.
# Does not wait for, or share a core with, the spin-boson gesvd jobs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
HERE="$ROOT/benchmark/JuliaVsC++/profiling"
JULIA_BIN="${JULIA_BIN:-julia}"
ACE_BIN="${ACE_BIN:-$ROOT/ACE/bin/ACE}"
PARAM="$ROOT/benchmark/JuliaVsC++/cpp/central_spin_partial.param"

echo "central-spin N=5 SVD audit; not waiting for spin-boson"

mapfile -t PICK < <(python3 - << 'PY'
import os
import time
def snap():
    out = {}
    with open("/proc/stat") as f:
        for line in f:
            if line.startswith("cpu") and line[3:4].isdigit():
                parts = line.split()
                cpu = int(parts[0][3:])
                nums = list(map(int, parts[1:]))
                idle = nums[3] + (nums[4] if len(nums) > 4 else 0)
                out[cpu] = (idle, sum(nums))
    return out
def siblings(cpu):
    with open(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list") as handle:
        text = handle.read().strip()
    vals = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            vals.extend(range(int(a), int(b) + 1))
        elif part:
            vals.append(int(part))
    return vals
def occupied():
    banned = set()
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\x00", b" ").decode(errors="replace")
            status = open(f"/proc/{pid}/status").read()
        except OSError:
            continue
        if "run_gesvd_case.jl" not in cmd and "ACE/bin/ACE" not in cmd:
            continue
        for line in status.splitlines():
            if not line.startswith("Cpus_allowed_list:"):
                continue
            for part in line.split(":", 1)[1].split(","):
                part = part.strip()
                if not part:
                    continue
                if "-" in part:
                    lo, hi = part.split("-")
                    banned.update(range(int(lo), int(hi) + 1))
                else:
                    banned.add(int(part))
    extra = set()
    for cpu in banned:
        extra.update(siblings(cpu))
    return banned | extra

a = snap(); time.sleep(0.4); b = snap()
banned = occupied()
def idle_frac(cpu):
    di = b[cpu][0] - a[cpu][0]
    dt = b[cpu][1] - a[cpu][1]
    return di / dt if dt else 0.0
chosen = None
for cpu in sorted(b):
    if cpu in banned or idle_frac(cpu) < 0.9:
        continue
    sibs = [s for s in siblings(cpu) if s != cpu]
    if sibs and any(s in banned or idle_frac(s) < 0.9 for s in sibs):
        continue
    chosen = cpu
    if cpu == 0:
        break
if chosen is None:
    raise SystemExit("no idle core with an idle sibling outside the spin-boson jobs")
print(chosen)
print(" ".join(str(c) for c in sorted(banned)))
PY
)
CPU="${PICK[0]}"
EXCLUDED="${PICK[1]:-}"
echo "pinning measurements to CPU $CPU (sibling idle; excluded $EXCLUDED)"

make -C "$HERE/cpp"
mkdir -p "$HERE/logs" "$HERE/results" "$HERE/fixtures"
cat > "$HERE/logs/isolation.txt" << EOF
case: partial central spin N=5
pinned_cpu: $CPU
excluded_cpus: $EXCLUDED
spin-boson gesvd jobs were left running on their own cores and were not waited on.
This audit does not share those cores or their sibling hyperthreads.
EOF
LIB="$HERE/cpp/libsvd_count.so"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MKL_DYNAMIC=FALSE OMP_DYNAMIC=FALSE
export PROFILING_CPU="$CPU"
export SVD_COUNT_LIB="$LIB"
export JULIA_BIN

run_pinned() {
  taskset -c "$CPU" "$@"
}

echo "=== phase 1A ==="
run_pinned env LD_PRELOAD="$LIB" \
  "$JULIA_BIN" +1.12.7 -t 1 --project="$ROOT/benchmark" "$HERE/run_phase1a.jl" \
  | tee "$HERE/logs/phase1a.log"

echo "=== phase 1B ==="
run_pinned env LD_PRELOAD= \
  "$JULIA_BIN" +1.12.7 -t 1 --project="$ROOT/benchmark" "$HERE/run_phase1b.jl" \
  | tee "$HERE/logs/phase1b.log"

echo "=== C++ N=5 gesvd count ==="
: > "$HERE/logs/cpp_zgesvd.csv"
set +e
(
  cd "$HERE/logs"
  run_pinned env LD_PRELOAD="$LIB" SVD_COUNT_LOG="$HERE/logs/cpp_zgesvd.csv" \
    "$ACE_BIN" "$PARAM" \
    -RandomSpin_N_modes 5 \
    -RandomSpin_J_max 0.2 \
    -RandomSpin_J_min 0.2 \
    -dont_propagate true \
    -write_PT "$HERE/logs/cpp_partial_N5.pt" \
    -RandomSpin_print_initial "$HERE/logs/cpp_partial_N5_orientations.txt"
) > "$HERE/logs/cpp_partial_N5.log" 2>&1
cpp_status=$?
set -e
echo "C++ ACE exit $cpp_status" | tee -a "$HERE/logs/cpp_partial_N5.log"
if [[ "$cpp_status" -ne 0 ]]; then
  echo "C++ count run failed; continuing to the report with the Julia measurements" >&2
fi

echo "=== phase 1C ==="
run_pinned env LD_PRELOAD= \
  "$JULIA_BIN" +1.12.7 -t 1 --project="$ROOT/benchmark" "$HERE/run_phase1c.jl" \
  | tee "$HERE/logs/phase1c.log"

echo "audit complete: $HERE/report.md"
