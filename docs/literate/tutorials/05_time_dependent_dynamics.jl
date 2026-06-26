# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors #src
# SPDX-License-Identifier: MIT #src
# #src
# File: docs/literate/tutorials/05_time_dependent_dynamics.jl #src
# Contributor: Gauthameshwar S. #src
# #src
# Literate tutorial source: piecewise time-dependent driving (placeholder). #src

# # Time-Dependent Dynamics
#
# This tutorial will show how to approximate time-dependent Hamiltonians by
# stitching together `tebd` steps with updated `OpSum` generators.

using ProcessTensors
using ITensors.Ops: Trotter

sites = siteinds("S=1/2", 1)
H_drive_x = OpSum() + (1.0, "Sx", 1)
H_drive_z = OpSum() + (1.0, "Sz", 1)
ψ = random_mps(sites; linkdims=4)
ψ = tebd(ψ, H_drive_x, 0.05, 0.05; alg=Trotter{2}())
ψ = tebd(ψ, H_drive_z, 0.05, 0.05; alg=Trotter{2}())

@assert ψ isa MPS{Hilbert}
@assert isapprox(norm(ψ), 1.0; atol=1e-10)

# ## Planned sections
#
# - Piecewise-constant driving with repeated `tebd` calls
# - Error scaling with time step
# - Manual time dependence for `tdvp` workflows
