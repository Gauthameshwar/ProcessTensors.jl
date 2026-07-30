# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: scripts/terminal/evolve.jl
# Contributor: Gauthameshwar S.
#
# Demonstrates reduced-state trajectory progress and persistent verbose logging
# for a small single-mode spin-bath process tensor.
#
# Run with:
# julia --project=. scripts/terminal/evolve.jl

include("common.jl")
@info "Loaded common.jl"

progress = true
verbose = true
problem = terminal_spin_bath_problem()

@info "Building process tensor..."
pt = build_process_tensor(
    problem.system;
    environment=problem.bath,
    dt=problem.dt,
    nsteps=problem.nsteps,
    progress=true,
)

@info "Evolving process..."
trajectory = evolve(pt, problem.rho0; progress=progress, verbose=verbose)
println("Computed $(length(trajectory.states_hilbert)) reduced snapshots.")
