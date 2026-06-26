# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors #src
# SPDX-License-Identifier: MIT #src
# #src
# File: docs/literate/tutorials/04_dissipative_dynamics.jl #src
# Contributor: Gauthameshwar S. #src
# #src
# Literate tutorial source: dissipative Liouville dynamics (placeholder). #src

# # Dissipative Dynamics
#
# This tutorial will introduce Lindblad evolution on `MPS{Liouville}` using
# `tebd` and `tdvp`, including jump operators and Trotter error discussion.

using ProcessTensors
using ITensors.Ops: Trotter

sites = siteinds("S=1/2", 1)
s_L = liouv_sites(sites)
H = OpSum()
H += 1.0, "Sz", 1
ρ0 = to_liouville(to_dm(MPS(sites, "Up")); sites=s_L)
ρT = tebd(ρ0, H, 0.1, 0.1; jump_ops=[(0.1, "S-", 1)], alg=Trotter{2}())
Sz = MPO(OpSum() + (1.0, "Sz", 1), sites)

@assert ρT isa MPS{Liouville}
sz0 = real(tr(apply(Sz, to_hilbert(ρ0); alg="naive", truncate=false)))
szT = real(tr(apply(Sz, to_hilbert(ρT); alg="naive", truncate=false)))
@assert szT < sz0

# ## Planned sections
#
# - Master equation and `OpSum_Liouville`
# - `tebd` / `tdvp` on vectorized density matrices
# - Validation against small exact references
