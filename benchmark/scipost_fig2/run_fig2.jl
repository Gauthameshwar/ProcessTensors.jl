# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# Generate ACE temporal-bond data for SciPost Figure 2.
#
# Run with:
#   julia -t auto --project=benchmark benchmark/scipost_fig2/run_fig2.jl

include(joinpath(@__DIR__, "common.jl"))

const SEED = parse(Int, get(ENV, "SCIPOST_FIG2_SEED", string(DEFAULT_SEED)))
const N_BATH = parse(Int, get(ENV, "SCIPOST_FIG2_N", string(DEFAULT_N_BATH)))
const J = parse(Float64, get(ENV, "SCIPOST_FIG2_J", string(DEFAULT_J)))
const FINAL_TIME =
    parse(Float64, get(ENV, "SCIPOST_FIG2_T", string(DEFAULT_FINAL_TIME)))
const DT_REFERENCE =
    parse(Float64, get(ENV, "SCIPOST_FIG2_DT_REF", string(DEFAULT_DT_REFERENCE)))
const TIMESTEPS = parse_float_list(
    get(ENV, "SCIPOST_FIG2_DTS", join(DEFAULT_TIMESTEPS, ',')),
)
const CUTOFFS = parse_float_list(
    get(ENV, "SCIPOST_FIG2_CUTOFFS", join(DEFAULT_CUTOFFS, ',')),
)
const MAXDIM = parse(Int, get(ENV, "SCIPOST_FIG2_MAXDIM", string(DEFAULT_MAXDIM)))

function main()
    N_BATH >= 1 || throw(ArgumentError("SCIPOST_FIG2_N must be positive."))
    MAXDIM >= 1 || throw(ArgumentError("SCIPOST_FIG2_MAXDIM must be positive."))
    all(>(0), CUTOFFS) || throw(ArgumentError("All cutoffs must be positive."))

    timesteps = sort(unique(vcat(TIMESTEPS, DT_REFERENCE)); rev=true)
    for dt in timesteps
        nsteps_for(FINAL_TIME, dt)
    end

    generated = sample_bath_orientations(N_BATH, SEED)
    write_orientations(ORIENTATIONS_PATH, generated)
    orientations = read_orientations(ORIENTATIONS_PATH)
    length(orientations) == N_BATH || error(
        "Expected $N_BATH stored orientations, got $(length(orientations)).",
    )

    bath = unpolarized_central_spin_bath(orientations; J)
    central_spin = empty_central_spin_system()
    profile_rows = Any[]
    dmax_rows = Any[]

    println("SciPost Figure 2: ACE temporal bond dimensions")
    println("  N=$N_BATH  J=$J  T=$FINAL_TIME  seed=$SEED  RNG=Xoshiro")
    println("  dt=$(join(timesteps, ','))")
    println("  cutoff=$(join(CUTOFFS, ','))  maxdim=$MAXDIM  compression=canonzip")

    for dt in timesteps
        nsteps = nsteps_for(FINAL_TIME, dt)
        for cutoff in CUTOFFS
            println("ACE: dt=$dt, nsteps=$nsteps, cutoff=$cutoff")
            profile = build_ace_bond_profile(
                central_spin.system,
                bath;
                dt,
                nsteps,
                cutoff,
                maxdim=MAXDIM,
            )
            @printf(
                "  Dmax=%d  capped=%s  build=%.3f s\n",
                profile.dmax,
                profile.hit_maxdim,
                profile.build_seconds,
            )

            for k in 0:nsteps
                push!(
                    profile_rows,
                    (
                        profile.times[k + 1],
                        k,
                        dt,
                        nsteps,
                        cutoff,
                        profile.dimensions[k + 1],
                        profile.dmax,
                        profile.hit_maxdim,
                        profile.build_seconds,
                    ),
                )
            end
            push!(
                dmax_rows,
                (
                    dt,
                    nsteps,
                    cutoff,
                    profile.dmax,
                    profile.hit_maxdim,
                    profile.build_seconds,
                ),
            )
        end
    end

    write_csv(
        joinpath(RESULTS_DIR, "fig2_profiles.csv"),
        (
            "t", "k", "dt", "nsteps", "cutoff", "D_k", "D_max",
            "hit_maxdim", "t_build_s",
        ),
        profile_rows,
    )
    write_csv(
        joinpath(RESULTS_DIR, "fig2_dmax.csv"),
        ("dt", "nsteps", "cutoff", "D_max", "hit_maxdim", "t_build_s"),
        dmax_rows,
    )
    write_environment(
        ;
        seed=SEED,
        n_bath=N_BATH,
        J,
        final_time=FINAL_TIME,
        dt_reference=DT_REFERENCE,
        timesteps,
        cutoffs=CUTOFFS,
        maxdim=MAXDIM,
        orientations_path=ORIENTATIONS_PATH,
    )
    return nothing
end

main()
