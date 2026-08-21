# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: benchmark/ace_compressors/ace_compression_sanity.jl
# Contributor: Gauthameshwar S.
#
# Compares zip-up and canonzip ACE against Dense on a tiny two-mode
# noncommuting spin bath at ε = 0 and ε = 1e-10.
#
# Run with:
#   julia -t auto --project=. benchmark/ace_compressors/ace_compression_sanity.jl

include(joinpath(@__DIR__, "ace_compression_common.jl"))

const DT = 0.1
const NSTEPS = 6
const CUTOFFS = (0.0, 1e-10)

function main()
    println("ACE compression sanity")
    println("----------------------")
    sys_phys = siteinds("S=1/2", 1)
    system = with_logger(NullLogger()) do
        spin_system(sys_phys, OpSum() + (0.7, "Sx", 1))
    end
    bath = two_mode_noncommuting_bath()
    rho0 = to_dm(MPS(sys_phys, ["Up"]))

    pt_dense = with_logger(NullLogger()) do
        build_process_tensor(
            system;
            method=Dense(),
            environment=bath,
            dt=DT,
            nsteps=NSTEPS,
            sys_alg=Trotter{2}(),
            progress=false,
        )
    end
    traj_dense = trajectory_matrices(pt_dense, rho0)
    println("Dense reference built (physics only; ACE timings use BenchmarkTools).")

    header = (
        "strategy", "cutoff", "t_build", "t_median_s", "t_mean_s",
        "allocated_bytes", "allocated_mib", "allocs", "nsamples",
        "chi_max", "eps_rho", "eps_TP", "eps_H", "eps_pos",
    )
    rows = []
    trajs = Dict{Tuple{Symbol,Float64},Any}()

    for cutoff in CUTOFFS
        println()
        @printf("cutoff = %.1e\n", cutoff)
        for strategy in STRATEGIES
            pt, st = measure_ace_build(
                system, bath;
                cutoff=cutoff, compression=strategy, dt=DT, nsteps=NSTEPS,
            )
            traj = trajectory_matrices(pt, rho0)
            trajs[(strategy, cutoff)] = traj
            dash = worst_diagnostics(traj)
            eps_rho = max_traj_error(traj, traj_dense)
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
        if cutoff == 0.0
            e_zc = max_traj_error(trajs[(:zipup, 0.0)], trajs[(:canonzip, 0.0)])
            @printf("  pairwise |zipup-canonzip|_max = %.3e\n", e_zc)
        end
    end

    write_csv(results_path("ace_compression_sanity.csv"), header, rows)
    return nothing
end

main()
