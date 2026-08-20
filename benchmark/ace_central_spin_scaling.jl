# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: benchmark/ace_central_spin_scaling.jl
# Contributor: Gauthameshwar S.
#
# Scaling comparison of zip-up and canonzip ACE on the polarized central-spin
# model. Finite-N deviation from (1/2) cos(t/2) is physics, not the
# compression-error metric.
#
# Run with:
#   julia -t auto --project=. benchmark/ace_central_spin_scaling.jl
#   ACE_RUN_LARGE=true julia -t auto --project=. benchmark/ace_central_spin_scaling.jl

include(joinpath(@__DIR__, "ace_compression_common.jl"))

const J = 1.0
const DT = 0.05
const T_FINAL = 4.0
const NSTEPS = round(Int, T_FINAL / DT) + 1
const CUTOFF = 1e-10
const N_DEFAULT = [10, 25, 50, 100]
const N_LARGE = [10, 50, 100, 300, 1000]

function main()
    n_values = get(ENV, "ACE_RUN_LARGE", "false") == "true" ? N_LARGE : N_DEFAULT
    println("ACE central-spin scaling")
    println("------------------------")
    @printf("  dt = %.3f  t_final = %.1f  nsteps = %d  ε = %.1e\n", DT, T_FINAL, NSTEPS, CUTOFF)
    println("  N = $(join(n_values, ", "))")

    system, sites = empty_spin_system()
    rho0 = to_dm(MPS(sites, ["+"]))

    header = (
        "strategy", "N", "cutoff", "t_build", "t_median_s", "t_mean_s",
        "allocated_bytes", "allocated_mib", "allocs", "nsamples",
        "chi_max", "eps_analytic", "eps_TP", "eps_H", "eps_pos",
    )
    rows = []

    for N_bath in n_values
        println()
        println("N = $N_bath")
        bath = polarized_central_spin_bath(N_bath; J=J)
        for strategy in STRATEGIES
            pt, st = measure_ace_build(
                system, bath;
                cutoff=CUTOFF, compression=strategy, dt=DT, nsteps=NSTEPS,
            )
            traj = trajectory_matrices(pt, rho0)
            times = range(0.0, step=DT, length=NSTEPS)
            Sx = ComplexF64[0.0 0.5; 0.5 0.0]
            sx = [real(tr(Sx * ρ)) for ρ in traj]
            analytic = [0.5 * cos(t / 2) for t in times]
            dash = worst_diagnostics(traj)
            eps_analytic = maximum(abs.(sx .- analytic))
            χ = maxlinkdim(pt.core)
            @printf(
                "  %-10s  t_min=%.3f s  t_med=%.3f s  MiB=%.2f  n=%d  χ=%d  |Sx-cos|_max=%.3e\n",
                strategy, st.t_min_s, st.t_median_s, st.memory_bytes / 2^20, st.nsamples, χ, eps_analytic,
            )
            push!(
                rows,
                (
                    strategy, N_bath, CUTOFF, st.t_min_s, st.t_median_s, st.t_mean_s,
                    st.memory_bytes, st.memory_bytes / 2^20, st.allocs, st.nsamples,
                    χ, eps_analytic, dash.trace, dash.hermiticity, dash.positivity,
                ),
            )
        end
    end

    write_csv(results_path("ace_central_spin_scaling.csv"), header, rows)
    return nothing
end

main()
