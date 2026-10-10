# Shared provenance, pinning, and CSV helpers for the C++ ACE construction
# wrappers. Source this file; do not execute it.

# Prefer the wrapper's HERE (absolute JuliaVsC++ directory). Fall back to
# this file's location when sourced directly.
if [[ -n "${HERE:-}" && -d "$HERE" ]]; then
  BENCH_HERE="$HERE"
else
  _BENCH_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  BENCH_HERE="$(cd "$_BENCH_COMMON_DIR/.." && pwd)"
fi
if [[ -z "${ACE_ROOT:-}" ]]; then
  if [[ -n "${ROOT:-}" ]]; then
    ACE_ROOT="$ROOT/ACE"
  else
    ACE_ROOT="$(cd "$BENCH_HERE/../.." && pwd)/ACE"
  fi
fi
if [[ -z "${EIGEN_HOME:-}" && -d "$ACE_ROOT/external/eigen-3.4.0" ]]; then
  EIGEN_HOME="$ACE_ROOT/external/eigen-3.4.0"
fi

is_darwin() {
  [[ "$(uname -s)" == Darwin ]]
}

# macOS cannot pin a process to a core; CSV rows record this instead of a CPU id.
pin_label() {
  if is_darwin; then
    printf 'unpinned'
  else
    printf '%s' "$1"
  fi
}


single_thread_env() {
  export OMP_NUM_THREADS="${ACE_OMP_THREADS:-1}"
  export MKL_NUM_THREADS="${ACE_MKL_THREADS:-1}"
  export OPENBLAS_NUM_THREADS="${ACE_OPENBLAS_THREADS:-1}"
  export MKL_DYNAMIC=FALSE
  export OMP_DYNAMIC=FALSE
  widen_runner_affinity
}

# Machine-online CPUs, not the current process affinity. `nproc` follows the
# inherited taskset/cgroup mask and can report 1 even on a 128-thread host.
online_cpu_spec() {
  if is_darwin; then
    # Apple Silicon: default worker pool is the performance cores only, so
    # concurrent cases are not scheduled onto efficiency cores.
    local p
    p=$(sysctl -n hw.perflevel0.logicalcpu 2>/dev/null || sysctl -n hw.logicalcpu)
    printf '0-%s' "$((p - 1))"
  elif [[ -r /sys/devices/system/cpu/online ]]; then
    tr -d ' \n' < /sys/devices/system/cpu/online
  else
    printf '0-%s' "$(($(nproc --all) - 1))"
  fi
}

expand_cpu_spec() {
  local spec="$1" part a b i
  local -a out=()
  IFS=',' read -r -a parts <<< "$spec"
  for part in "${parts[@]}"; do
    part="${part// /}"
    [[ -z "$part" ]] && continue
    if [[ "$part" == *-* ]]; then
      a="${part%-*}"
      b="${part#*-}"
      for ((i = a; i <= b; i++)); do
        out+=("$i")
      done
    else
      out+=("$part")
    fi
  done
  printf '%s' "${out[*]}"
}

current_affinity_spec() {
  if is_darwin; then
    printf '0-%s' "$(($(sysctl -n hw.logicalcpu) - 1))"
    return
  fi
  awk '/^Cpus_allowed_list:/ { print $2; exit }' /proc/self/status 2>/dev/null || nproc
}

widen_runner_affinity() {
  is_darwin && return 0
  local spec current
  spec="$(online_cpu_spec)"
  current="$(current_affinity_spec)"
  if [[ "$current" == "$spec" ]]; then
    return 0
  fi
  if command -v taskset >/dev/null 2>&1 && taskset -cp "$spec" $$ >/dev/null 2>&1; then
    echo "Widened runner CPU affinity from $current to $spec"
    return 0
  fi
  echo "WARNING: inherited affinity is $current; could not widen to $spec." >&2
  echo "Workers may be unable to use every online core. Run from an unrestricted shell or set ACE_CPUS." >&2
}

default_cpu_list() {
  widen_runner_affinity
  local raw
  raw="${ACE_CPUS:-${JULIA_SPINBOSON_CPUS:-${JULIA_VS_CPP_CPUS:-}}}"
  if [[ -z "$raw" ]]; then
    raw="$(expand_cpu_spec "$(online_cpu_spec)")"
  fi
  read -r -a BENCH_CPU_LIST <<< "$raw"
  if (( ${#BENCH_CPU_LIST[@]} == 0 )); then
    echo "No CPUs listed in ACE_CPUS / JULIA_VS_CPP_CPUS." >&2
    return 1
  fi
  local duplicates
  duplicates=$(printf '%s\n' "${BENCH_CPU_LIST[@]}" | sort | uniq -d)
  if [[ -n "$duplicates" ]]; then
    echo "Duplicate CPU ids are not allowed (exclusive pinning): $duplicates" >&2
    return 1
  fi
}

need_cpu_list() {
  default_cpu_list
  if (( ${#BENCH_CPU_LIST[@]} < 1 )); then
    echo "Set ACE_CPUS or JULIA_VS_CPP_CPUS to at least one CPU id." >&2
    return 1
  fi
  echo "Using ${#BENCH_CPU_LIST[@]} CPUs as a pool for ${#cases[@]} cases (later cases wait for a free core)."
}

# Batches `cases` onto BENCH_CPU_LIST. Caller must define:
#   queue_launch spec cpu  -> sets q_pid q_log q_frag q_label and backgrounds the job
#   merge_fragment fragment
run_in_cpu_batches() {
  local ncpu=${#BENCH_CPU_LIST[@]} batch_start batch_end i slot
  local failed=0
  local -a pids logs frags labs
  for ((batch_start = 0; batch_start < ${#cases[@]}; batch_start += ncpu)); do
    batch_end=$((batch_start + ncpu))
    ((batch_end > ${#cases[@]})) && batch_end=${#cases[@]}
    pids=(); logs=(); frags=(); labs=()
    slot=0
    for ((i = batch_start; i < batch_end; i++)); do
      queue_launch "${cases[$i]}" "${BENCH_CPU_LIST[$slot]}"
      pids+=("$q_pid")
      logs+=("$q_log")
      frags+=("$q_frag")
      labs+=("$q_label")
      slot=$((slot + 1))
    done
    for i in "${!pids[@]}"; do
      if wait "${pids[$i]}"; then
        [[ -f "${logs[$i]}" ]] && cat "${logs[$i]}"
        merge_fragment "${frags[$i]}"
      else
        echo "Case failed: ${labs[$i]} (see ${logs[$i]})" >&2
        [[ -f "${logs[$i]}" ]] && tail -n 40 "${logs[$i]}" >&2
        failed=1
      fi
    done
    ((failed == 0)) || return 1
  done
}

require_exclusive_cpus() {
  need_cpu_list
}

ace_linked_libs() {
  local bin="$1"
  if is_darwin; then
    otool -L "$bin" 2>/dev/null
  else
    ldd "$bin" 2>/dev/null
  fi
}

ace_compiler_flags() {
  local flags="-O3 -fno-math-errno -g -m64 --std=c++14 -fPIC -pthread"
  local lib="$ACE_ROOT/lib/libACE.so"
  if [[ -n "${MKLROOT:-}" ]]; then
    flags="$flags -DEIGEN_USE_MKL_ALL"
  elif ace_linked_libs "$lib" | grep -qi openblas; then
    flags="$flags -DEIGEN_USE_BLAS -DEIGEN_USE_LAPACKE"
  fi
  printf '%s' "$flags"
}

# The compiler that built ACE: GCC links its own libstdc++, clang links libc++.
ace_compiler_version() {
  local lib="$ACE_ROOT/lib/libACE.so" gxx
  if [[ -n "${ACE_CXX:-}" ]]; then
    "$ACE_CXX" --version | head -n 1
  elif is_darwin && ace_linked_libs "$lib" | grep -q 'opt/gcc/'; then
    gxx=$(ls /opt/homebrew/bin/g++-[0-9]* 2>/dev/null | sort -V | tail -n 1)
    "$gxx" --version | head -n 1
  else
    ${CXX:-g++} --version | head -n 1
  fi
}

ace_eigen_version() {
  local root="${EIGEN_HOME:-/usr/include/eigen3}"
  local hdr
  for hdr in "$root/Eigen/Version" "$root/Eigen/src/Core/util/Macros.h"; do
    [[ -f "$hdr" ]] || continue
    awk '
      /^#define EIGEN_WORLD_VERSION/ {w=$3}
      /^#define EIGEN_MAJOR_VERSION/ {j=$3}
      /^#define EIGEN_MINOR_VERSION/ {n=$3}
      /^#define EIGEN_PATCH_VERSION/ {p=$3}
      END {
        if (w == "") exit 1
        if (p != "") printf "%s.%s.%s", j, n, p; else printf "%s.%s.%s", w, j, n
      }
    ' "$hdr" && printf ' (%s)' "$root" && return
  done
  printf 'unknown'
}

ace_openblas_version() {
  local cfg="${OPENBLAS_HOME:-/opt/homebrew/opt/openblas}/include/openblas_config.h"
  if [[ -f "$cfg" ]]; then
    sed -n 's/^#define OPENBLAS_VERSION " *\(.*[^ ]\) *"/\1/p' "$cfg"
  else
    printf 'OpenBLAS (version unknown)'
  fi
}

ace_blas_link() {
  local lib="$ACE_ROOT/lib/libACE.so"
  if ace_linked_libs "$lib" | grep -qi mkl; then
    printf 'Intel MKL (linked)'
  elif ace_linked_libs "$lib" | grep -qi openblas; then
    printf '%s via LAPACKE; no MKL; Eigen JacobiSVD -> LAPACKE_?gesvd' "$(ace_openblas_version)"
  elif [[ -n "${MKLROOT:-}" ]]; then
    printf 'MKLROOT set but ACE binary is not linked to MKL'
  else
    printf 'no MKL, no LAPACK (Eigen built-in JacobiSVD)'
  fi
}

ace_cpu_model() {
  if is_darwin; then
    sysctl -n machdep.cpu.brand_string
  else
    awk -F': ' '/model name/ {print $2; exit}' /proc/cpuinfo 2>/dev/null || echo unknown
  fi
}

ace_ram() {
  if is_darwin; then
    awk -v b="$(sysctl -n hw.memsize)" 'BEGIN { printf "%.1f GiB", b / 1024 / 1024 / 1024 }'
  else
    awk '/MemTotal:/ {printf "%.1f GiB", $2/1024/1024}' /proc/meminfo 2>/dev/null || echo unknown
  fi
}

ace_commit() {
  if [[ -d "$ACE_ROOT/.git" ]]; then
    git -C "$ACE_ROOT" rev-parse --short HEAD 2>/dev/null || printf 'unknown'
  else
    printf 'not a git checkout'
  fi
}

print_timing_policy() {
  cat <<'EOF'
TIMING POLICY
  This is the Julia–C++ construction comparison. dont_propagate=true, so
  contract_s=0 and total_s is the ACE process wall time (setup + build +
  write_PT I/O). Compare Julia construction with build_s. I/O from
  PTB_analyze is recorded separately and is not charged to ACE.
  Table S.3.1 complete-example times live only in the sanity folders.

EOF
}

print_ace_provenance() {
  local bin="${ACE_BIN:-$ACE_ROOT/bin/ACE}"
  local cpu_model cores ram host cores_all
  cpu_model=$(ace_cpu_model)
  if is_darwin; then
    cores_all=$(sysctl -n hw.logicalcpu)
  else
    cores_all=$(nproc --all 2>/dev/null || nproc)
  fi
  cores=$(expand_cpu_spec "$(current_affinity_spec)" | wc -w | tr -d ' ')
  ram=$(ace_ram)
  host="$(uname -s) $(uname -r) $(uname -m)"
  cat <<EOF
ACE C++ BENCHMARK PROVENANCE
ACE commit           : $(ace_commit)
compiler/version     : $(ace_compiler_version)
compile flags        : $(ace_compiler_flags)
Eigen version        : $(ace_eigen_version)
BLAS/LAPACK          : $(ace_blas_link)
OMP threads          : ${OMP_NUM_THREADS:-1}
MKL threads          : ${MKL_NUM_THREADS:-1}
OPENBLAS threads     : ${OPENBLAS_NUM_THREADS:-1}
CPU model            : $cpu_model
logical CPUs         : $cores_all
cpus available       : $cores
cpus allowed         : $(current_affinity_spec)
available RAM        : $ram
host/kernel          : $host
ACE binary           : $bin
CPU pinning          : $(is_darwin && echo 'none (macOS has no core affinity; pool = performance cores)' || echo 'taskset -c, one exclusive CPU per case')
timing clock         : $(is_darwin && echo '/usr/bin/time -p real seconds' || echo 'GNU time %e wall seconds')
timing scope         : setup / build / contraction / I/O / total

EOF
}

print_run_parameters() {
  cat <<EOF
RUN PARAMETERS
run_id               : $1
model                : $2
bath_state           : $3
N_modes              : $4
local_dim            : $5
polarization         : $6
dt                   : $7
t_final              : $8
nsteps               : $9
cutoff               : ${10}
symmetric_trotter    : ${11}
dont_propagate       : true
seed                 : ${12}
temperature          : ${13}
spectral_parameters  : ${14}
pinned_cpu           : ${15}
OMP/MKL/OpenBLAS     : ${OMP_NUM_THREADS:-1}/${MKL_NUM_THREADS:-1}/${OPENBLAS_NUM_THREADS:-1}

EOF
}

csv_feq() {
  awk -v a="$1" -v b="$2" 'BEGIN {
    da = a + 0; db = b + 0
    diff = da > db ? da - db : db - da
    scale = (da > db ? da : db)
    if (scale < 1) scale = 1
    exit !(diff / scale < 1e-8)
  }'
}

ratio_or_blank() {
  local published="$1" localt="$2"
  if [[ -z "$published" || -z "$localt" ]]; then
    printf ''
    return
  fi
  awk -v p="$published" -v l="$localt" 'BEGIN {
    if ((l + 0) == 0) { print ""; exit }
    printf "%.6f", (p + 0) / (l + 0)
  }'
}

print_results_block() {
  cat <<EOF
RESULTS
D_max                : $1
setup_s              : $2
build_s              : $3
contract_s           : $4
io_s                 : $5
total_s              : $6
peak_rss_kb          : $7
swap_io              : $8
timing_scope         : $9

EOF
}

print_validation() {
  cat <<EOF
VALIDATION
max_bond_dimension   : $1
final_trace_error    : $2
trajectory_checksum  : $3
max_reference_error  : $4
status               : $5

EOF
}

# Rewrite an existing CSV so its header matches $2 and missing fields are blank.
ensure_csv() {
  local file="$1" header="$2"
  if [[ ! -s "$file" ]]; then
    printf '%s\n' "$header" > "$file"
    return
  fi
  local old
  old=$(head -n 1 "$file")
  [[ "$old" == "$header" ]] && return
  local tmp
  tmp=$(mktemp)
  awk -F, -v header="$header" '
    BEGIN {
      n = split(header, h, ",")
    }
    NR == 1 {
      for (i = 1; i <= NF; i++) old[$i] = i
      print header
      next
    }
    {
      elapsed = ""
      for (i = 1; i <= n; i++) {
        col = h[i]
        val = (col in old && old[col] <= NF) ? $old[col] : ""
        v[i] = val
        if (col == "elapsed_sec" || col == "t_min_s") {
          if (elapsed == "" && val != "") elapsed = val
        }
      }
      for (i = 1; i <= n; i++) {
        col = h[i]
        if ((col == "build_s" || col == "total_s") && v[i] == "" && elapsed != "") v[i] = elapsed
        if ((col == "contract_s" || col == "io_s") && v[i] == "") v[i] = "0"
      }
      out = ""
      for (i = 1; i <= n; i++) {
        out = out v[i] (i < n ? "," : "")
      }
      print out
    }
  ' "$file" > "$tmp"
  mv "$tmp" "$file"
}

parse_gnu_time() {
  # Sets elapsed_sec, peak_rss_kb, swap_io from a GNU time -f '%e %M %W' file.
  local timef="$1"
  read -r elapsed_sec peak_rss_kb swap_io < "$timef"
  elapsed_sec=${elapsed_sec:-}
  peak_rss_kb=${peak_rss_kb:-}
  swap_io=${swap_io:-}
}

contract_from_log() {
  local log="$1"
  local ms
  ms=$(sed -n 's/.*runtime for propagation: \([0-9.]*\)ms.*/\1/p' "$log" | tail -n 1)
  if [[ -z "$ms" ]]; then
    printf '0'
  else
    awk -v ms="$ms" 'BEGIN { printf "%.6f", ms / 1000.0 }'
  fi
}

# Run ACE pinned to one CPU. Arguments after the first four are passed to ACE.
# Sets: elapsed_sec peak_rss_kb swap_io contract_s
run_ace_pinned() {
  local cpu="$1" log="$2" timef="$3"
  shift 3
  single_thread_env
  if is_darwin; then
    local raw="$timef.raw"
    /usr/bin/time -l -p -o "$raw" "$ACE_BIN" "$@" > "$log" 2>&1
    awk '
      $1 == "real" { e = $2 }
      /maximum resident set size/ { m = int($1 / 1024) }
      $2 == "swaps" { w = $1 }
      END { printf "%s %s %s\n", e, m, w }
    ' "$raw" > "$timef"
    rm -f "$raw"
  else
    local -a cmd=()
    if [[ -n "$cpu" ]]; then
      cmd+=(taskset -c "$cpu")
    fi
    cmd+=("$ACE_BIN" "$@")
    /usr/bin/time -f '%e %M %W' -o "$timef" "${cmd[@]}" > "$log" 2>&1
  fi
  parse_gnu_time "$timef"
  contract_s=$(contract_from_log "$log")
}

now_s() {
  if is_darwin; then
    perl -MTime::HiRes=time -e 'printf "%.6f", time'
  else
    date +%s.%N
  fi
}

time_cmd() {
  # Print elapsed seconds of a command to stdout; command output goes to $1.
  local log="$1"
  shift
  local start end
  start=$(now_s)
  "$@" >> "$log" 2>&1 || return $?
  end=$(now_s)
  awk -v s="$start" -v e="$end" 'BEGIN { printf "%.6f", e - s }'
}
