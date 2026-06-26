# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors #src
# SPDX-License-Identifier: MIT #src
# #src
# File: docs/literate/tutorials/02_liouville_basics.jl #src
# Contributor: Gauthameshwar S. #src
# #src
# Literate tutorial source: Liouville-space basics (placeholder). #src

# # Liouville Basics
#
# This tutorial will explain vectorized density matrices, `liouv_sites`, and
# round-trips between Hilbert MPOs and Liouville MPS objects.

using ProcessTensors

s = siteinds("S=1/2", 1)
s_L = liouv_sites(s)
ρ = to_dm(MPS(s, "Up"))
ρL = to_liouville(ρ; sites=s_L)
ρ_back = to_hilbert(ρL)

@assert ρL isa MPS{Liouville}
@assert ρ ≈ ρ_back

# ## Planned sections
#
# - `liouv_sites` and index reuse
# - `to_liouville` / `to_hilbert` and `combiners`
# - Building `OpSum_Liouville` and `MPO_Liouville`
