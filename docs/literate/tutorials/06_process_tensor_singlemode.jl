# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors #src
# SPDX-License-Identifier: MIT #src
# #src
# File: docs/literate/tutorials/06_process_tensor_singlemode.jl #src
# Contributor: Gauthameshwar S. #src
# #src
# Literate tutorial source: single-mode process tensor (placeholder). #src

# # Single-Mode Process Tensor
#
# This tutorial will build a process tensor with one bath mode and extract
# reduced system trajectories with `evolve`.

using ProcessTensors

sys = siteinds("S=1/2", 1)
env = siteinds("S=1/2", 1)
L_sys = liouv_sites(sys)
L_env = liouv_sites(env)

H_sys = OpSum() + (0.3, "Sz", 1)
system = spin_system(sys, H_sys)

ρ_env = to_liouville(to_dm(MPS(env, "Up")); sites=L_env)
H_env = OpSum() + (1.0, "Sx", 1)
coupling = OpSum() + (1.0, "Sz", 1, "Sz", 2)
mode = spin_mode(L_env, H_env, ρ_env; coupling=coupling)
bath = spin_bath([mode])

pt = build_process_tensor(system, only(system.sites); environment=bath, dt=0.05, nsteps=3)
ρ0 = to_dm(MPS(sys, "Up"))
trj = evolve(pt, ρ0)

@assert length(trj.states_hilbert) == pt.nsteps
@assert trj.states_hilbert[end] isa MPO{Hilbert}

# ## Planned sections
#
# - Process-tensor legs and embedded system propagation
# - `InstrumentSeq`, `StatePreparation`, and `TraceOut`
# - Markovian limit vs bath-induced memory
