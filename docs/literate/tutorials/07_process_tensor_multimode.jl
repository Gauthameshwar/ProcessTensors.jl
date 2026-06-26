# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors #src
# SPDX-License-Identifier: MIT #src
# #src
# File: docs/literate/tutorials/07_process_tensor_multimode.jl #src
# Contributor: Gauthameshwar S. #src
# #src
# Literate tutorial source: multimode process tensor (placeholder). #src

# # Multimode Process Tensor
#
# This tutorial will construct a process tensor from multiple bath modes and
# discuss Liouville dimension limits for exact system-bath evolution.

using ProcessTensors

sys = siteinds("S=1/2", 1)
e1 = siteinds("S=1/2", 1)
e2 = siteinds("S=1/2", 1)
L_sys = liouv_sites(sys)
L1 = liouv_sites(e1)
L2 = liouv_sites(e2)

system = spin_system(sys, OpSum() + (0.2, "Sz", 1))
H_env = OpSum() + (1.0, "Sx", 1)

ρ1 = to_liouville(to_dm(MPS(e1, "Up")); sites=L1)
ρ2 = to_liouville(to_dm(MPS(e2, "Up")); sites=L2)
m1 = spin_mode(L1, H_env, ρ1; coupling=OpSum() + (0.05, "Sz", 1, "Sz", 2))
m2 = spin_mode(L2, H_env, ρ2; coupling=OpSum() + (0.03, "Sz", 1, "Sz", 2))
bath = spin_bath([m1, m2])

pt = build_process_tensor(system, only(system.sites); environment=bath, dt=0.05, nsteps=2)
trj = evolve(pt, to_dm(MPS(sys, "Up")))

@assert length(pt.core) == 2
@assert length(trj.states_liouville) == 2

# ## Planned sections
#
# - Stacking several `SpinMode` / `BosonicMode` objects
# - Joint Liouville dimension and `MAX_DENSE_LIOUVILLE_DIM`
# - Reduced observables along multimode memory
