# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors #src
# SPDX-License-Identifier: MIT #src
# #src
# File: docs/literate/tutorials/03_unitary_dynamics.jl #src
# Contributor: Gauthameshwar S. #src
# #src
# Literate tutorial source: unitary TEBD/TDVP dynamics (placeholder). #src

# # Unitary Dynamics
#
# This tutorial will cover Hilbert-space time evolution with `tebd` and `tdvp`,
# Trotter factorization, and comparison with density-matrix workflows.

using ProcessTensors
using ITensors.Ops: Trotter

sites = siteinds("S=1/2", 2)
H = OpSum()
H += 0.5, "Sz", 1, "Sz", 2
ψ0 = random_mps(sites; linkdims=4)
ψT = tebd(ψ0, H, 0.1, 0.2; alg=Trotter{2}())

@assert ψT isa MPS{Hilbert}
@assert isapprox(norm(ψT), norm(ψ0); atol=1e-10)

# ## Planned sections
#
# - `tebd` on `MPS{Hilbert}` and `Trotter{n}()` choices
# - `tdvp` wrapper semantics
# - Pure-state vs density-matrix evolution on small chains
