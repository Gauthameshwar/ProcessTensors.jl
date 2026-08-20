# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: benchmark/ace_compression_common.jl
# Contributor: Gauthameshwar S.
#
# Shared helpers for ACE zip-up vs canonzip compression benchmarks. Writes CSV
# output under benchmark/results/. Runtime and memory come from BenchmarkTools
# after warmup, not a single compiling @elapsed.

import Pkg

const _BENCH_ENV = joinpath(@__DIR__, ".bench_env")
const _ORIG_PROJECT = Base.active_project()
mkpath(_BENCH_ENV)
Pkg.activate(_BENCH_ENV)
if !isfile(joinpath(_BENCH_ENV, "Manifest.toml"))
    Pkg.add("BenchmarkTools")
else
    Pkg.instantiate()
end
using BenchmarkTools
Pkg.activate(_ORIG_PROJECT)

using LinearAlgebra
using Printf
using Statistics
using ProcessTensors
using ITensors
using ITensors.Ops: Trotter
using Logging

const STRATEGIES = (:zipup, :canonzip)
const RESULTS_DIR = joinpath(@__DIR__, "results")
const BENCH_SAMPLES = parse(Int, get(ENV, "ACE_BENCH_SAMPLES", "5"))
const BENCH_SECONDS = parse(Float64, get(ENV, "ACE_BENCH_SECONDS", "1800"))

function results_path(name::AbstractString)
    mkpath(RESULTS_DIR)
    return joinpath(RESULTS_DIR, name)
end

function write_csv(path::AbstractString, header, rows)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(header, ","))
        for row in rows
            println(io, join(row, ","))
        end
    end
    println("Wrote $path")
    return path
end

function one_site_density_matrix(ρ)
    T = foldl(*, ρ)
    site = only(filter(i -> plev(i) == 0 && hastags(i, "Site"), inds(T)))
    return ComplexF64.(Array(T, prime(site), site))
end

function density_diagnostics(ρ::AbstractMatrix)
    trace_error = abs(tr(ρ) - 1)
    nrm = max(norm(ρ), eps())
    hermiticity = norm(ρ - ρ') / nrm
    λmin = minimum(real, eigvals(Hermitian((ρ + ρ') / 2)))
    positivity = max(0.0, -λmin)
    return (trace=trace_error, hermiticity=hermiticity, positivity=positivity)
end

function trajectory_matrices(pt, rho0)
    trajectory = evolve(pt, rho0; progress=false)
    return [one_site_density_matrix(ρ) for ρ in trajectory.states_hilbert]
end

function max_traj_error(traj_a, traj_b)
    return maximum(norm(a - b) for (a, b) in zip(traj_a, traj_b))
end

function worst_diagnostics(matrices)
    traces = Float64[]
    herms = Float64[]
    poss = Float64[]
    for ρ in matrices
        d = density_diagnostics(ρ)
        push!(traces, d.trace)
        push!(herms, d.hermiticity)
        push!(poss, d.positivity)
    end
    return (
        trace=maximum(traces),
        hermiticity=maximum(herms),
        positivity=maximum(poss),
    )
end

function trial_stats(trial)
    return (
        t_min_s=minimum(trial).time / 1e9,
        t_median_s=median(trial).time / 1e9,
        t_mean_s=mean(trial).time / 1e9,
        memory_bytes=Int(memory(trial)),
        allocs=Int(allocs(trial)),
        nsamples=length(trial.times),
    )
end

function measure_ace_build(system, bath; cutoff, compression, dt, nsteps, maxdim=typemax(Int))
    pt = build_ace_pt(
        system, bath;
        cutoff=cutoff, compression=compression, dt=dt, nsteps=nsteps, maxdim=maxdim,
    )
    trial = benchmark_ace_build(
        system, bath;
        cutoff=cutoff, compression=compression, dt=dt, nsteps=nsteps, maxdim=maxdim,
    )
    return pt, trial_stats(trial)
end

function benchmark_ace_build(
    system,
    bath;
    cutoff,
    compression,
    dt,
    nsteps,
    maxdim=typemax(Int),
    samples=BENCH_SAMPLES,
    seconds=BENCH_SECONDS,
)
    bench = @benchmarkable build_ace_pt(
        $system,
        $bath;
        cutoff=$cutoff,
        compression=$compression,
        dt=$dt,
        nsteps=$nsteps,
        maxdim=$maxdim,
    )
    return run(bench; samples=samples, evals=1, seconds=seconds)
end

function build_ace_pt(system, bath; cutoff, compression, dt, nsteps, maxdim=typemax(Int))
    return with_logger(NullLogger()) do
        build_process_tensor(
            system;
            method=ACE(cutoff=cutoff, maxdim=maxdim, compression=compression),
            environment=bath,
            dt=dt,
            nsteps=nsteps,
            sys_alg=Trotter{2}(),
            combine_alg=Trotter{2}(),
            progress=false,
        )
    end
end

function spin_mode_on_axis(h, g, axis::AbstractString, init::AbstractString)
    env_phys = siteinds("S=1/2", 1)
    env_liouv = liouv_sites(env_phys)
    rho_env = to_liouville(to_dm(MPS(env_phys, [init])); sites=env_liouv)
    H_mode = OpSum() + (h, "Sx", 1)
    cpl = OpSum() + (g, axis, 1, axis, 2)
    return with_logger(NullLogger()) do
        spin_mode(env_liouv, H_mode, rho_env; coupling=cpl)
    end
end

function two_mode_noncommuting_bath()
    return with_logger(NullLogger()) do
        spin_bath([
            spin_mode_on_axis(0.3, 0.04, "Sz", "Up"),
            spin_mode_on_axis(0.35, 0.05, "Sx", "Up"),
        ])
    end
end

function heterogeneous_eight_spin_bath()
    specs = (
        (0.40, 0.12, "Sz", "Up"),
        (0.55, 0.18, "Sx", "+"),
        (0.30, 0.09, "Sy", "Dn"),
        (0.70, 0.15, "Sz", "-"),
        (0.22, 0.07, "Sx", "Up"),
        (0.48, 0.20, "Sy", "+"),
        (0.61, 0.11, "Sz", "Dn"),
        (0.35, 0.14, "Sx", "-"),
    )
    modes = [spin_mode_on_axis(h, g, ax, init) for (h, g, ax, init) in specs]
    return with_logger(NullLogger()) do
        spin_bath(modes)
    end
end

function polarized_spin_density(physical_site, liouville_site)
    return to_liouville(
        to_dm(MPS([physical_site], ["Up"]));
        sites=[liouville_site],
    )
end

function polarized_central_spin_bath(N_bath::Int; J::Real=1.0)
    Jk = J / N_bath
    bath_sites = siteinds("S=1/2", N_bath)
    bath_liouville_sites = liouv_sites(bath_sites)
    modes = SpinMode[]
    with_logger(NullLogger()) do
        for k in 1:N_bath
            coupling = OpSum()
            coupling += Jk, "Sx", 1, "Sx", 2
            coupling += Jk, "Sy", 1, "Sy", 2
            coupling += Jk, "Sz", 1, "Sz", 2
            push!(
                modes,
                spin_mode(
                    [bath_liouville_sites[k]],
                    OpSum(),
                    polarized_spin_density(bath_sites[k], bath_liouville_sites[k]);
                    coupling=coupling,
                ),
            )
        end
        return spin_bath(modes)
    end
end

function empty_spin_system()
    sites = siteinds("S=1/2", 1)
    system = with_logger(NullLogger()) do
        spin_system(sites, OpSum())
    end
    return system, sites
end
