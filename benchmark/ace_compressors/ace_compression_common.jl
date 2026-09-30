# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: benchmark/ace_compressors/ace_compression_common.jl
# Contributor: Gauthameshwar S.
#
# Shared helpers for ACE compression-schedule benchmarks. Writes CSV output
# under benchmark/ace_compressors/results/. Runtime and memory come from
# BenchmarkTools after warmup, not a single compiling @elapsed.

include(joinpath(@__DIR__, "..", "env.jl"))

using BenchmarkTools

using LinearAlgebra
using Printf
using Random
using Statistics
using ProcessTensors
using ITensors
using ITensors.Ops: Trotter
using Logging

const STRATEGIES = (:zipup_cpp, :canonzip)
const RESULTS_DIR = joinpath(@__DIR__, "results")
const ORIENTATION_SEED = 20260905
const ORIENTATIONS_PATH = joinpath(RESULTS_DIR, "bath_orientations.csv")
const BENCH_SAMPLES = parse(Int, get(ENV, "ACE_BENCH_SAMPLES", "5"))
const BENCH_SECONDS = parse(Float64, get(ENV, "ACE_BENCH_SECONDS", "1800"))

function rotated_strategies(case_index::Integer)
    shift = mod(case_index - 1, length(STRATEGIES))
    return circshift(STRATEGIES, -shift)
end

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

"""Maximum Frobenius distance between two evolved trajectories."""
function max_traj_error(traj_a, traj_b)
    length(traj_a) == length(traj_b) || throw(
        ArgumentError("Trajectories have different lengths: $(length(traj_a)) vs $(length(traj_b))."),
    )
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

"""Draw `n_bath` pure-spin orientations uniformly on the Bloch sphere."""
function sample_bath_orientations(n_bath::Int, seed::Integer=ORIENTATION_SEED)
    n_bath >= 1 || throw(ArgumentError("n_bath must be positive; got $n_bath."))
    rng = Xoshiro(seed)
    return map(1:n_bath) do k
        u = rand(rng)
        v = rand(rng)
        nz = 1 - 2u
        θ = acos(clamp(nz, -1, 1))
        ϕ = 2π * v
        sinθ = sin(θ)
        (; k, theta=θ, phi=ϕ, nx=sinθ * cos(ϕ), ny=sinθ * sin(ϕ), nz)
    end
end

function write_orientations(path::AbstractString, orientations)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "k,theta,phi,nx,ny,nz")
        for orientation in orientations
            @printf(
                io,
                "%d,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                orientation.k,
                orientation.theta,
                orientation.phi,
                orientation.nx,
                orientation.ny,
                orientation.nz,
            )
        end
    end
    println("Wrote $path")
    return path
end

function bloch_spin_density(physical_site, liouville_site, orientation)
    θ = orientation.theta
    ϕ = orientation.phi
    amplitudes = ComplexF64[cos(θ / 2), cis(ϕ) * sin(θ / 2)]
    ket = MPS(ITensor(amplitudes, physical_site), [physical_site])
    return to_liouville(to_dm(ket); sites=[liouville_site])
end

"""Cygorek b=0 unpolarised central-spin bath for one archived realization."""
function unpolarized_central_spin_bath(orientations; J::Real=1.0)
    n_bath = length(orientations)
    n_bath >= 1 || throw(ArgumentError("Need at least one bath orientation."))
    bath_sites = siteinds("S=1/2", n_bath)
    bath_liouville_sites = liouv_sites(bath_sites)
    Jk = J / n_bath
    modes = SpinMode[]
    with_logger(NullLogger()) do
        for k in 1:n_bath
            coupling = OpSum()
            coupling += Jk, "Sx", 1, "Sx", 2
            coupling += Jk, "Sy", 1, "Sy", 2
            coupling += Jk, "Sz", 1, "Sz", 2
            push!(
                modes,
                spin_mode(
                    [bath_liouville_sites[k]],
                    OpSum(),
                    bloch_spin_density(
                        bath_sites[k],
                        bath_liouville_sites[k],
                        orientations[k],
                    );
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
