# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: benchmark/evolve_contractors/evolve_contraction.jl
# Contributor: Gauthameshwar S.
#
# Compares `:evaluate` and `:closures` evolve contractions on synthetic
# random process tensors with prescribed bond dimension. Reports median runtime
# and memory versus nsteps and χ. Cores are random; only index structure and
# contraction cost matter.
#
# Run with:
#   julia -t auto --project=. benchmark/evolve_contractors/evolve_contraction.jl

include(joinpath(@__DIR__, "common.jl"))

import ITensorMPS
using Random

const DT = 0.05
const RNG_SEED = 1
const NSTEPS_SWEEP = [16, 32, 64, 128, 256]
const NSTEPS_FOR_CHI = 32
const CHI_FOR_NSTEPS = 8
const CHI_SWEEP = [4, 8, 16, 32, 64, 128, 256]
const CONTRACTIONS = (:evaluate, :closures)
const EVOLVE_BENCH_SAMPLES = parse(Int, get(ENV, "EVOLVE_BENCH_SAMPLES", "3"))
const EVOLVE_BENCH_SECONDS = parse(Float64, get(ENV, "EVOLVE_BENCH_SECONDS", "1800"))

function random_itensor_on(rng::AbstractRNG, inds)
    data = randn(rng, ComplexF64, map(dim, inds)...)
    T = ITensor(data, inds...)
    nrm = norm(T)
    nrm > 0 || return T
    return T / nrm
end

function random_process_tensor(nsteps::Integer, chi::Integer; dt::Real=DT, seed::Integer=RNG_SEED)
    nsteps >= 1 || throw(ArgumentError("random_process_tensor: nsteps must be ≥ 1."))
    chi >= 1 || throw(ArgumentError("random_process_tensor: chi must be ≥ 1."))
    rng = Random.Xoshiro(seed)
    system, _ = empty_spin_system()
    coupling_site = only(system.sites)
    links = [Index(chi; tags="Link,tstep=$k") for k in 1:(nsteps - 1)]

    cores = ITensor[]
    for k in 0:(nsteps - 1)
        in_k, out_k = ProcessTensors._generate_pt_legs(coupling_site, k)
        inds = Index[in_k, out_k]
        k > 0 && push!(inds, links[k])
        k < nsteps - 1 && push!(inds, links[k + 1])
        push!(cores, random_itensor_on(rng, Tuple(inds)))
    end
    return ProcessTensor(ITensorMPS.MPO(cores), system, nothing, dt, nsteps, coupling_site)
end

function contraction_name(contraction)
    return string(contraction)
end

function measure_evolve(pt, rho0, contraction; samples=EVOLVE_BENCH_SAMPLES, seconds=EVOLVE_BENCH_SECONDS)
    bench = @benchmarkable evolve(
        $pt,
        $rho0;
        contraction=$contraction,
        progress=false,
    )
    trial = run(bench; samples=samples, evals=1, seconds=seconds)
    return trial_stats(trial)
end

function snapshot_error(pt, rho0)
    traj_eval = [
        one_site_density_matrix(ρ) for ρ in
        evolve(pt, rho0; contraction=:evaluate, progress=false).states_hilbert
    ]
    traj_close = [
        one_site_density_matrix(ρ) for ρ in
        evolve(pt, rho0; contraction=:closures, progress=false).states_hilbert
    ]
    return maximum(
        norm(a - b) / max(norm(a), norm(b), eps())
        for (a, b) in zip(traj_eval, traj_close)
    )
end

function print_row(label, χ, nsteps, st, eps)
    @printf(
        "  %-20s  nsteps=%-3d  χ=%-4d  t_med=%.4f s  MiB=%.2f  n=%d  |Δρ|_max=%.3e\n",
        label, nsteps, χ, st.t_median_s, st.memory_bytes / 2^20, st.nsamples, eps,
    )
    return nothing
end

function run_point!(rows, rho0; nsteps, chi)
    println()
    println("nsteps = $nsteps  χ = $chi")
    pt = random_process_tensor(nsteps, chi; seed=RNG_SEED)
    χ = maxlinkdim(pt.core)
    χ == chi || @warn "realized χ=$χ differs from requested chi=$chi"
    eps = snapshot_error(pt, rho0)
    println("  random PT  χ=$(χ)  |ρ_eval - ρ_close|_max=$(eps)")

    for contraction in CONTRACTIONS
        st = measure_evolve(pt, rho0, contraction)
        name = contraction_name(contraction)
        print_row(name, χ, nsteps, st, eps)
        push!(
            rows,
            (
                name, nsteps, chi, χ, RNG_SEED,
                st.t_min_s, st.t_median_s, st.t_mean_s,
                st.memory_bytes, st.memory_bytes / 2^20, st.allocs, st.nsamples, eps,
            ),
        )
    end
    return nothing
end

function main()
    println("evolve contraction benchmark (synthetic random PT)")
    println("-------------------------------------------------")
    println("  seed = $RNG_SEED  dt = $DT")
    println("  nsteps sweep: $(join(NSTEPS_SWEEP, ", "))  at χ = $CHI_FOR_NSTEPS")
    println("  χ sweep: $(join(CHI_SWEEP, ", "))  at nsteps = $NSTEPS_FOR_CHI")

    _, sites = empty_spin_system()
    rho0 = to_dm(MPS(sites, ["+"]))

    header = (
        "contraction", "nsteps", "chi", "chi_max", "seed",
        "t_min_s", "t_median_s", "t_mean_s",
        "allocated_bytes", "allocated_mib", "allocs", "nsamples", "eps_eval_vs_close",
    )
    rows = []

    println()
    println("Sweep 1: nsteps at χ = $CHI_FOR_NSTEPS")
    for nsteps in NSTEPS_SWEEP
        run_point!(rows, rho0; nsteps=nsteps, chi=CHI_FOR_NSTEPS)
    end

    println()
    println("Sweep 2: χ at nsteps = $NSTEPS_FOR_CHI")
    for chi in CHI_SWEEP
        nsteps = NSTEPS_FOR_CHI
        nsteps == NSTEPS_FOR_CHI && chi == CHI_FOR_NSTEPS && continue
        run_point!(rows, rho0; nsteps=nsteps, chi=chi)
    end

    write_csv(results_path("evolve_contraction.csv"), header, rows)
    return nothing
end

main()
