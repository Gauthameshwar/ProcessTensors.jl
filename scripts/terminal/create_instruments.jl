# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: scripts/terminal/create_instruments.jl
# Contributor: Gauthameshwar S.
#
# Demonstrates direct instrument-schedule materialization with interactive
# progress and persistent verbose logging.
#
# Run with:
# julia --project=. scripts/terminal/create_instruments.jl

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
    progress=progress,
)

@info "Built process tensor. Creating instruments..."

seq = default_schedule(pt)
add!(seq, StatePreparation(problem.rho0), 0)
add!(seq, TraceOut(), pt.nsteps)
instruments = create_instruments(pt, seq; progress=progress, verbose=verbose)

println("Materialized $(length(instruments)) schedule slots.")
