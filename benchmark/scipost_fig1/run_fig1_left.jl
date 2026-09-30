# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# Generate timestep-convergence data for SciPost Figure 1(a,b).
#
# Run with:
#   OPENBLAS_NUM_THREADS=1 julia -t auto --project=benchmark benchmark/scipost_fig1/run_fig1_left.jl

include(joinpath(@__DIR__, "common.jl"))

const FINAL_TIME = parse(Float64, get(ENV, "SCIPOST_FIG1_T", string(DEFAULT_FINAL_TIME)))
const TIMESTEPS = parse_float_list(get(ENV, "SCIPOST_FIG1_DTS", "0.20,0.10,0.05"))
const ACE_CUTOFF = parse(Float64, get(ENV, "SCIPOST_FIG1_CUTOFF", string(DEFAULT_ACE_CUTOFF)))

function main()
    write_environment()
    model = multimode_spin_model()
    rows = Any[]

    for dt in TIMESTEPS
        nsteps = nsteps_for(FINAL_TIME, dt)
        times = collect(range(0.0; step=dt, length=nsteps + 1))
        sy_ed = direct_ed(model, times)

        println("Dense exact PT: dt=$dt, nsteps=$nsteps, T=$FINAL_TIME")
        dense = exact_pt_trajectory(model; dt, nsteps, sys_order=2)
        println("ACE: dt=$dt, nsteps=$nsteps, cutoff=$ACE_CUTOFF")
        ace = ace_trajectory(
            model; dt, nsteps, sys_order=2, mode_order=2, cutoff=ACE_CUTOFF,
        )
        dense.times == times || error("Dense PT time grid does not match ED.")
        ace.times == times || error("ACE time grid does not match ED.")
        @printf(
            "  max |⟨σy⟩−ED|: Dense=%.3e  ACE=%.3e\n",
            maximum(abs.(dense.sy .- sy_ed)),
            maximum(abs.(ace.sy .- sy_ed)),
        )

        for k in eachindex(times)
            push!(
                rows,
                (
                    times[k], dt, nsteps, "ed", sy_ed[k], 0.0,
                    0, NaN, "NA", "NA", 0.0,
                ),
            )
            push!(
                rows,
                (
                    dense.times[k], dt, nsteps, "exact_pt", dense.sy[k],
                    abs(dense.sy[k] - sy_ed[k]), dense.chi_max, NaN,
                    "Trotter2", "NA", dense.build_seconds,
                ),
            )
            push!(
                rows,
                (
                    ace.times[k], dt, nsteps, "ace", ace.sy[k],
                    abs(ace.sy[k] - sy_ed[k]), ace.chi_max, ACE_CUTOFF,
                    "Trotter2", "Trotter2", ace.build_seconds,
                ),
            )
        end
    end

    header = (
        "t", "dt", "nsteps", "method", "sy", "eps_sy", "chi_max",
        "cutoff", "sys_alg", "combine_alg", "t_build_s",
    )
    write_csv(joinpath(RESULTS_DIR, "fig1_left.csv"), header, rows)
    return nothing
end

main()
