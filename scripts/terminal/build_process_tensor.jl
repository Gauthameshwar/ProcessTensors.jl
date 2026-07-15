# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: scripts/terminal/build_process_tensor.jl
# Contributor: Gauthameshwar S.
#
# Demonstrates interactive process-tensor construction progress and persistent
# verbose logging for a small single-mode spin bath.
#
# Run with:
# julia --project=. scripts/terminal/build_process_tensor.jl

include("common.jl")
@info "Loaded common.jl"

# Try `progress=:auto, verbose=false` for normal interactive use, or
# `progress=false, verbose=true` to see headless-style durable logs only.
progress = true
verbose = true

problem = terminal_spin_bath_problem()

@info "Building process tensor..."
pt = build_process_tensor(
    problem.system;
    environment=problem.bath,
    dt=problem.dt,
    nsteps=problem.nsteps,
    alg=Exact(),
    sys_alg=Trotter{2}(),
    progress=progress,
    verbose=verbose,
)

println(pt)
