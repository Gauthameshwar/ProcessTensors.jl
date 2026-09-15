# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: benchmark/ace_compressors/ace_central_spin_scaling.jl
# Contributor: Gauthameshwar S.
#
# Scaling comparison of zip-up, C++-schedule zip-up, and canonzip ACE on
# Cygorek's unpolarised (b = 0) central-spin model. One archived Bloch
# realization is nested so that each N uses the first N bath spins.
#
# Run with:
#   julia -t auto --project=. benchmark/ace_compressors/ace_central_spin_scaling.jl

include(joinpath(@__DIR__, "ace_compression_common.jl"))

const J = 1.0
const DT = 0.05
const T_FINAL = 4.0
const NSTEPS = round(Int, T_FINAL / DT) + 1
const CUTOFF = 1e-10
const N_VALUES = [1, 5, 10, 20, 30, 40, 50, 70]

function main()
    n_values = N_VALUES
    n_max = maximum(n_values)
    orientations = sample_bath_orientations(n_max, ORIENTATION_SEED)
    write_orientations(ORIENTATIONS_PATH, orientations)

    println("ACE unpolarised central-spin scaling")
    println("------------------------------------")
    @printf("  dt = %.3f  t_final = %.1f  nsteps = %d  ε = %.1e\n", DT, T_FINAL, NSTEPS, CUTOFF)
    println("  N = $(join(n_values, ", "))")
    println("  seed = $ORIENTATION_SEED  orientations = $ORIENTATIONS_PATH")
    println("  Julia threads = $(Threads.nthreads())")
    println("  BLAS threads  = $(BLAS.get_num_threads())")

    system, sites = empty_spin_system()
    rho0 = to_dm(MPS(sites, ["+"]))

    header = (
        "strategy", "N", "cutoff", "t_build", "t_median_s", "t_mean_s",
        "allocated_bytes", "allocated_mib", "allocs", "nsamples",
        "chi_max", "eps_trace", "eps_H", "eps_pos",
    )
    rows = []

    for (case_index, N_bath) in enumerate(n_values)
        println()
        println("N = $N_bath")
        bath = unpolarized_central_spin_bath(orientations[1:N_bath]; J=J)
        strategy_order = rotated_strategies(case_index)
        println("strategy order = $strategy_order")
        for strategy in strategy_order
            pt, st = measure_ace_build(
                system, bath;
                cutoff=CUTOFF, compression=strategy, dt=DT, nsteps=NSTEPS,
            )
            traj = trajectory_matrices(pt, rho0)
            dash = worst_diagnostics(traj)
            χ = maxlinkdim(pt.core)
            @printf(
                "  %-10s  t_min=%.3f s  t_med=%.3f s  MiB=%.2f  n=%d  χ=%d\n",
                strategy, st.t_min_s, st.t_median_s, st.memory_bytes / 2^20, st.nsamples, χ,
            )
            push!(
                rows,
                (
                    strategy, N_bath, CUTOFF, st.t_min_s, st.t_median_s, st.t_mean_s,
                    st.memory_bytes, st.memory_bytes / 2^20, st.allocs, st.nsamples,
                    χ, dash.trace, dash.hermiticity, dash.positivity,
                ),
            )
        end
    end

    write_csv(results_path("ace_central_spin_scaling.csv"), header, rows)
    return nothing
end

main()
