# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: scripts/terminal/evaluate_process.jl
# Contributor: Gauthameshwar S.
#
# Demonstrates process evaluation with instrument materialization, contraction
# progress, and durable verbose open-leg diagnostics.
#
# Run with:
# julia --project=. scripts/terminal/evaluate_process.jl

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

Sz = OpSum() + (1.0, "Sz", 1)
final_sites = output_sites(pt, pt.nsteps - 1)
seq = default_schedule(pt)
add!(seq, StatePreparation(problem.rho0), 0)
# add!(seq, ObservableMeasurement(Sz, final_sites), pt.nsteps)

@info "Evaluating process..."
value = evaluate_process(pt, seq; progress=progress, verbose=verbose)
# println("Final ⟨Sᶻ⟩ = ", real(value))
