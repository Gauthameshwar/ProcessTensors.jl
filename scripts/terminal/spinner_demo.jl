# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: scripts/terminal/spinner_demo.jl
# Contributor: Gauthameshwar S.
#
# Demonstrates the standard ProcessTensors.jl progress-reporting interface on
# a deterministic dummy workflow. The sleeps intentionally make each stage
# visible; real package algorithms should report actual scientific work.
#
# Run with:
# julia --project=. scripts/terminal/spinner_demo.jl

using ProcessTensors

"""
    dummy_tracked_process(; nsteps=30, dt=0.1, progress=:auto, verbose=false)

Run a deterministic dummy workflow that demonstrates:

- an indeterminate preparation stage;
- an indeterminate construction stage;
- a known-length progress bar;
- compact live values for simulation time and a mock bond dimension;
- final transient-display cleanup;
- persistent stage logs when `verbose=true`.

This function is a template for integrating progress into real
`ProcessTensors.jl` workflows.
"""
function dummy_tracked_process(;
    nsteps::Integer=30,
    dt::Real=0.1,
    progress::Union{Bool,Symbol}=:auto,
    verbose::Bool=false,
)
    nsteps > 0 || throw(ArgumentError("nsteps must be positive; got $nsteps."))
    dt > 0 || throw(ArgumentError("dt must be positive; got $dt."))

    run = ProcessTensors.@progress_start progress verbose "Running spinner demo" (
        nsteps=Int(nsteps),
        dt=dt,
    )

    try
        ProcessTensors.@progress_stage run "Preparing dummy workspace" (
            workspace_size=64,
        )
        sleep(2.0)

        ProcessTensors.@progress_stage run "Constructing dummy propagator"
        sleep(2.4)

        values = Vector{Float64}(undef, nsteps)
        max_bond_dimension = 1

        ProcessTensors.@progress_bar run "Advancing dummy trajectory" nsteps begin
            for step in 1:nsteps
                # Stand-in for one meaningful scientific timestep.
                sleep(0.15)

                physical_time = step * dt
                values[step] = sin(physical_time) * exp(-0.05 * physical_time)
                max_bond_dimension = min(256, max(max_bond_dimension, 2 * step))

                ProcessTensors.@progress_update run step (
                    time=round(physical_time; digits=2),
                    maxlinkdim=max_bond_dimension,
                )
            end
        end

        ProcessTensors.@progress_stage run "Finalizing dummy result" (
            final_maxlinkdim=max_bond_dimension,
        )
        sleep(1.0)

        return (
            times=dt .* collect(1:nsteps),
            values=values,
            final_maxlinkdim=max_bond_dimension,
        )
    finally
        ProcessTensors.@progress_finish run
    end
end

# Recommended combinations:
#
# Interaction-first local run:
#   progress=true,  verbose=false
#
# Headless or remote run:
#   progress=false, verbose=true
#
# Debug mode:
#   progress=true,  verbose=true
#
# Performance-first run:
#   progress=false, verbose=false

progress = true
verbose = false

result = dummy_tracked_process(
    nsteps=30,
    dt=0.1;
    progress=progress,
    verbose=verbose,
)

println()
println("Spinner demo completed.")
println("Snapshots: ", length(result.values))
println("Final mock bond dimension: ", result.final_maxlinkdim)
