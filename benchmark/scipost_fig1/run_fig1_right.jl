# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# Generate ACE propagation-order data for SciPost Figure 1(c,d).
#
# Run with:
#   OPENBLAS_NUM_THREADS=1 julia -t auto --project=benchmark benchmark/scipost_fig1/run_fig1_right.jl

include(joinpath(@__DIR__, "common.jl"))

const FINAL_TIME = parse(Float64, get(ENV, "SCIPOST_FIG1_T", string(DEFAULT_FINAL_TIME)))
const TIMESTEP = parse(Float64, get(ENV, "SCIPOST_FIG1_RIGHT_DT", "0.10"))
const ACE_CUTOFF = parse(Float64, get(ENV, "SCIPOST_FIG1_CUTOFF", string(DEFAULT_ACE_CUTOFF)))
const PROPAGATION_ORDERS = ((1, 1), (1, 2), (2, 1), (2, 2))

function main()
    write_environment()
    model = multimode_spin_model()
    nsteps = nsteps_for(FINAL_TIME, TIMESTEP)
    times = collect(range(0.0; step=TIMESTEP, length=nsteps + 1))
    sy_ed = direct_ed(model, times)
    rows = Any[]

    for k in eachindex(times)
        push!(
            rows,
            (
                times[k], TIMESTEP, nsteps, "ed", sy_ed[k], 0.0,
                0, 0, 0, NaN, 0.0,
            ),
        )
    end

    for (sys_order, mode_order) in PROPAGATION_ORDERS
        println(
            "ACE: dt=$TIMESTEP, nsteps=$nsteps, " *
            "sys_order=$sys_order, mode_order=$mode_order, cutoff=$ACE_CUTOFF",
        )
        ace = ace_trajectory(
            model;
            dt=TIMESTEP,
            nsteps,
            sys_order,
            mode_order,
            cutoff=ACE_CUTOFF,
        )
        ace.times == times || error("ACE time grid does not match ED.")
        @printf("  max |⟨σy⟩−ED| = %.3e\n", maximum(abs.(ace.sy .- sy_ed)))
        for k in eachindex(times)
            push!(
                rows,
                (
                    ace.times[k], TIMESTEP, nsteps, "ace", ace.sy[k],
                    abs(ace.sy[k] - sy_ed[k]), sys_order, mode_order,
                    ace.chi_max, ACE_CUTOFF, ace.build_seconds,
                ),
            )
        end
    end

    header = (
        "t", "dt", "nsteps", "method", "sy", "eps_sy", "sys_order",
        "mode_order", "chi_max", "cutoff", "t_build_s",
    )
    write_csv(joinpath(RESULTS_DIR, "fig1_right.csv"), header, rows)
    return nothing
end

main()
