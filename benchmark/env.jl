# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# Shared instantiate for `julia --project=benchmark`. Skip in parallel
# workers with SKIP_INSTANTIATE=1 after the parent has already instantiated.

using Pkg

if get(ENV, "SKIP_INSTANTIATE", "0") != "1"
    Pkg.instantiate()
end
