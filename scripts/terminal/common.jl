# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: scripts/terminal/common.jl
# Contributor: Gauthameshwar S.
#
# Defines a small single-spin bath problem shared by interactive terminal
# progress and logging demonstrations.

using ProcessTensors
using ITensors
using ITensors.Ops: Exact, Trotter

function terminal_spin_bath_problem(; dt=0.05, nsteps=24, nenv=5)
    sys_phys = siteinds("S=1/2", 1)
    env_phys = [siteinds("S=1/2", 1) for _ in 1:nenv]
    env_liouv = [liouv_sites(env_phys[i]) for i in 1:nenv]

    H_sys = OpSum() + (0.8, "Sx", 1)
    system = spin_system(sys_phys, H_sys)
    rho_env = [to_liouville(to_dm(MPS(env_phys[i], ["Up"])); sites=env_liouv[i]) for i in 1:nenv]
    H_env = [OpSum() + (0.5, "Sx", 1) for _ in 1:nenv]
    coupling = [OpSum() + (0.6, "Sz", 1, "Sz", 2) for i in 1:nenv]
    bath = spin_bath([SpinMode(sites=env_liouv[i], H=H_env[i], rho0=rho_env[i], coupling=coupling[i]) for i in 1:nenv])
    rho0 = to_dm(MPS(sys_phys, ["Up"]))

    return (; system, bath, rho0, dt, nsteps)
end
