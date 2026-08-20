# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: benchmark/ace_compression_benchmark.jl
# Contributor: Gauthameshwar S.
#
# ACE compression benchmark: compares zip-up and canonzip compression across
# several SVD cutoffs using an eight-spin environment, measuring process
# accuracy, bond dimension, runtime, and memory allocation. The reference is
# canonzip ACE at ε = 1e-13.
#
# Run with:
#   julia -t auto --project=. benchmark/ace_compression_benchmark.jl

include(joinpath(@__DIR__, "ace_compression_common.jl"))

const DT = 0.1
const NSTEPS = 8
const CUTOFFS = (1e-6, 1e-8, 1e-10, 1e-12)
const REFERENCE_CUTOFF = 1e-13

function main()
    println("ACE compression benchmark")
    println("-------------------------")
    sys_phys = siteinds("S=1/2", 1)
    system = with_logger(NullLogger()) do
        spin_system(sys_phys, OpSum() + (0.45, "Sx", 1) + (0.20, "Sz", 1))
    end
    bath = heterogeneous_eight_spin_bath()
    rho0 = to_dm(MPS(sys_phys, ["+"]))

    println("Building canonzip reference at ε = $REFERENCE_CUTOFF")
    pt_ref, st_ref = measure_ace_build(
        system, bath;
        cutoff=REFERENCE_CUTOFF, compression=:canonzip, dt=DT, nsteps=NSTEPS,
    )
    traj_ref = trajectory_matrices(pt_ref, rho0)
    dash_ref = worst_diagnostics(traj_ref)
    @printf(
        "reference  t_min=%.3f s  t_med=%.3f s  MiB=%.2f  n=%d  χ=%d\n",
        st_ref.t_min_s, st_ref.t_median_s, st_ref.memory_bytes / 2^20,
        st_ref.nsamples, maxlinkdim(pt_ref.core),
    )

    header = (
        "strategy", "cutoff", "t_build", "t_median_s", "t_mean_s",
        "allocated_bytes", "allocated_mib", "allocs", "nsamples",
        "chi_max", "eps_rho", "eps_TP", "eps_H", "eps_pos",
    )
    rows = Any[
        (
            :canonzip_ref, REFERENCE_CUTOFF, st_ref.t_min_s, st_ref.t_median_s, st_ref.t_mean_s,
            st_ref.memory_bytes, st_ref.memory_bytes / 2^20, st_ref.allocs, st_ref.nsamples,
            maxlinkdim(pt_ref.core), 0.0,
            dash_ref.trace, dash_ref.hermiticity, dash_ref.positivity,
        ),
    ]

    for cutoff in CUTOFFS
        println()
        @printf("cutoff = %.1e\n", cutoff)
        for strategy in STRATEGIES
            pt, st = measure_ace_build(
                system, bath;
                cutoff=cutoff, compression=strategy, dt=DT, nsteps=NSTEPS,
            )
            traj = trajectory_matrices(pt, rho0)
            dash = worst_diagnostics(traj)
            eps_rho = max_traj_error(traj, traj_ref)
            χ = maxlinkdim(pt.core)
            @printf(
                "  %-10s  t_min=%.3f s  t_med=%.3f s  MiB=%.2f  n=%d  χ=%d  ερ=%.3e\n",
                strategy, st.t_min_s, st.t_median_s, st.memory_bytes / 2^20, st.nsamples, χ, eps_rho,
            )
            push!(
                rows,
                (
                    strategy, cutoff, st.t_min_s, st.t_median_s, st.t_mean_s,
                    st.memory_bytes, st.memory_bytes / 2^20, st.allocs, st.nsamples,
                    χ, eps_rho, dash.trace, dash.hermiticity, dash.positivity,
                ),
            )
        end
    end

    write_csv(results_path("ace_compression_benchmark.csv"), header, rows)
    return nothing
end

main()
