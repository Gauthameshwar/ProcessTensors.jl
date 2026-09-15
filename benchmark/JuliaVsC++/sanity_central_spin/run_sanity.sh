#!/usr/bin/env bash
# Run the Julia/C++ central-spin comparison against exact diagonalization.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
ACE="$ROOT/ACE"

if [[ ! -x "$ACE/bin/ACE" ]]; then
  echo "ACE binary not found at $ACE/bin/ACE" >&2
  echo "Build it first: cd ACE && make" >&2
  exit 1
fi

SANITY_MODELS=central_spin exec bash "$HERE/../run_sanity_cpp_julia_ed.sh"
