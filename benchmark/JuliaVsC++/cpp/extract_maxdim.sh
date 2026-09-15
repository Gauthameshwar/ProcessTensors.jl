# Shared helper for the C++ ACE wrappers. Source this file; do not execute it.
# Prints the last outgoing construction Maxdim, or PTB_analyze's Maxdim.

maxdim_from_log() {
  local log="$1"
  local maxdim
  maxdim=$(sed -n 's/.*Maxdim at n=[0-9][0-9]*:.*-> *\([0-9][0-9]*\).*/\1/p' "$log" | tail -n 1)
  if [[ -z "$maxdim" ]]; then
    maxdim=$(sed -n 's/^Maxdim \([0-9][0-9]*\) at .*/\1/p' "$log" | tail -n 1)
  fi
  printf '%s\n' "$maxdim"
}
