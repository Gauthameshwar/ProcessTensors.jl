# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# Shared model helpers for the Julia vs C++ ACE construction benchmark.
# BenchmarkTools lives in benchmark/.bench_env and is not a package dependency.

import Pkg

const _BENCH_ROOT = dirname(@__DIR__)
const _BENCH_ENV = joinpath(_BENCH_ROOT, ".bench_env")
const _ORIG_PROJECT = Base.active_project()
mkpath(_BENCH_ENV)
Pkg.activate(_BENCH_ENV)
if !isfile(joinpath(_BENCH_ENV, "Manifest.toml"))
    Pkg.add(["BenchmarkTools", "Parsers"])
elseif get(ENV, "JULIA_VS_CPP_SKIP_INSTANTIATE", "0") != "1"
    Pkg.instantiate()
    deps = Pkg.project().dependencies
    haskey(deps, "Parsers") || Pkg.add("Parsers")
end
Pkg.activate(_ORIG_PROJECT)
if !(_BENCH_ENV in LOAD_PATH)
    pushfirst!(LOAD_PATH, _BENCH_ENV)
end
using BenchmarkTools

using Dates
using ITensors
using ITensors.Ops: Exact, Trotter
using LinearAlgebra
using Logging
using ProcessTensors
using Printf
using Random
using Statistics

include(joinpath(@__DIR__, "provenance.jl"))

const RESULTS_DIR = joinpath(@__DIR__, "results", "julia")
const CPP_RESULTS_DIR = joinpath(@__DIR__, "results", "cpp")
const BENCH_SAMPLES = parse(Int, get(ENV, "ACE_BENCH_SAMPLES", "1"))
const BENCH_SECONDS = parse(Float64, get(ENV, "ACE_BENCH_SECONDS", "1800"))

# Central spin (hbar = 0.6582119569 eV·ps)
const CS_J = 1 / 0.6582119569
const CS_FINAL_TIME = 20.0
const CS_DT = 0.1
const CS_CUTOFF = 1e-10
const CS_NS = (5, 10, 25, 50, 100)
const CS_SEED = 20260905
const CS_PARTIAL_B = (0.0, 0.0, 20.0)
const CS_PARTIAL_T_K = 11.604522
const CS_KB_MEV = 8.6173303e-2

# Spin-boson / independent-boson (Omega = 1)
const SB_OMEGA = 1.0
const SB_C = 0.2
const SB_GAMMA = 0.1
const SB_OMEGA_C = 1.0
const SB_OMEGA_MIN = 0.0
const SB_OMEGA_MAX = 7.5
const SB_N_MODES = 100
const SB_NS = (5, 10, 20, 50, 100)
const SB_LOCAL_DIM = 5
const SB_FINAL_TIME = 8.0
const SB_DT = 0.1
const SB_CUTOFF = 1e-8
const SB_TEMPERATURES = (0.0, 0.5, 1.0, 3.0)
const SB_MAXDIM = 4096
const CS_MAXDIM = 4096

nsteps_for(final_time::Real, dt::Real) = round(Int, final_time / dt)

function ensure_results_dir()
    mkpath(RESULTS_DIR)
    return RESULTS_DIR
end

function bloch_spin_density(physical_site, liouville_site, nx, ny, nz)
    n = hypot(nx, ny, nz)
    n > 0 || throw(ArgumentError("Bloch vector must be nonzero."))
    nx, ny, nz = nx / n, ny / n, nz / n
    θ = acos(clamp(nz, -1, 1))
    ϕ = atan(ny, nx)
    amplitudes = ComplexF64[cos(θ / 2), cis(ϕ) * sin(θ / 2)]
    ket = MPS(ITensor(amplitudes, physical_site), [physical_site])
    return to_liouville(to_dm(ket); sites=[liouville_site])
end

function sample_uniform_orientations(n_bath::Int, seed::Integer)
    rng = Xoshiro(seed)
    return map(1:n_bath) do _
        u = rand(rng)
        v = rand(rng)
        nz = 1 - 2u
        θ = acos(clamp(nz, -1, 1))
        ϕ = 2π * v
        sinθ = sin(θ)
        (nx=sinθ * cos(ϕ), ny=sinθ * sin(ϕ), nz)
    end
end

function sample_partial_orientations(n_bath::Int, seed::Integer)
    rng = Xoshiro(seed)
    Bx, By, Bz = CS_PARTIAL_B
    Bnorm = hypot(Bx, By, Bz)
    β = 1 / (CS_KB_MEV * CS_PARTIAL_T_K)
    orientations = NamedTuple{(:nx, :ny, :nz),Tuple{Float64,Float64,Float64}}[]
    while length(orientations) < n_bath
        u = rand(rng)
        v = rand(rng)
        nz = 1 - 2u
        θ = acos(clamp(nz, -1, 1))
        ϕ = 2π * v
        sinθ = sin(θ)
        nx, ny = sinθ * cos(ϕ), sinθ * sin(ϕ)
        E = 0.5 * (Bx * nx + By * ny + Bz * nz - Bnorm)
        if rand(rng) <= exp(β * E)
            push!(orientations, (; nx, ny, nz))
        end
    end
    return orientations
end

function polarised_orientations(n_bath::Int)
    return fill((nx=0.0, ny=0.0, nz=1.0), n_bath)
end

function cpp_orientation_path(polarisation::AbstractString, N::Int)
    return joinpath(CPP_RESULTS_DIR, "central_$(polarisation)_N$(N)_orientations.txt")
end

function parse_cpp_orientations(path::AbstractString)
    isfile(path) || throw(ArgumentError("C++ orientation file not found: $path"))
    orientations = NamedTuple{(:nx, :ny, :nz),Tuple{Float64,Float64,Float64}}[]
    for raw in eachline(path)
        line = strip(raw)
        isempty(line) && continue
        startswith(line, "#") && continue
        colon = findfirst(':', line)
        colon === nothing && throw(ArgumentError("Malformed C++ orientation line: $line"))
        fields = split(strip(line[(colon + 1):end]))
        length(fields) == 3 || throw(ArgumentError("Expected three Bloch components: $line"))
        push!(
            orientations,
            (
                nx=parse(Float64, fields[1]),
                ny=parse(Float64, fields[2]),
                nz=parse(Float64, fields[3]),
            ),
        )
    end
    isempty(orientations) && throw(ArgumentError("No orientations in $path"))
    return orientations
end

function total_environment_spin(orientations)
    sx = 0.5 * sum(o.nx for o in orientations)
    sy = 0.5 * sum(o.ny for o in orientations)
    sz = 0.5 * sum(o.nz for o in orientations)
    return (sx, sy, sz)
end

"""Load bath orientations for one C++ case.

Polarised baths are deterministic (+z). Partial and unpolarised baths must
reuse the C++ `*_orientations.txt` files so Julia compresses the same spins.
"""
function orientations_for(polarisation::AbstractString, N::Int)
    path = cpp_orientation_path(polarisation, N)
    if polarisation == "polarised"
        if isfile(path)
            orientations = parse_cpp_orientations(path)
            length(orientations) == N || throw(ArgumentError(
                "Polarised C++ file $path has $(length(orientations)) spins, expected $N.",
            ))
            return orientations
        end
        return polarised_orientations(N)
    end
    if polarisation == "partial" || polarisation == "unpolarised"
        orientations = parse_cpp_orientations(path)
        length(orientations) == N || throw(ArgumentError(
            "C++ file $path has $(length(orientations)) spins, expected $N.",
        ))
        return orientations
    end
    throw(ArgumentError("unknown polarisation: $polarisation"))
end

function central_spin_bath(orientations; J::Real=CS_J)
    n_bath = length(orientations)
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
            o = orientations[k]
            push!(
                modes,
                spin_mode(
                    [bath_liouville_sites[k]],
                    OpSum(),
                    bloch_spin_density(
                        bath_sites[k],
                        bath_liouville_sites[k],
                        o.nx,
                        o.ny,
                        o.nz,
                    );
                    coupling,
                ),
            )
        end
        return spin_bath(modes)
    end
end

function empty_central_spin_system()
    sites = siteinds("S=1/2", 1)
    system = with_logger(NullLogger()) do
        spin_system(sites, OpSum())
    end
    return system
end

lorentzian_J(ω; C=SB_C, γ=SB_GAMMA, ωc=SB_OMEGA_C) =
    (C / π) * γ / ((ω - ωc)^2 + γ^2)

function lorentzian_mode_grid(
    N=SB_N_MODES;
    ωmin=SB_OMEGA_MIN,
    ωmax=SB_OMEGA_MAX,
    C=SB_C,
)
    Δω = (ωmax - ωmin) / N
    frequencies = [ωmin + (k - 0.5) * Δω for k in 1:N]
    couplings = sqrt.(lorentzian_J.(frequencies; C) .* Δω)
    return frequencies, couplings
end

function thermal_boson_density(physical_site, liouville_site, ω, kBT, local_dim)
    occupations = 0:(local_dim - 1)
    if kBT <= 0
        weights = [n == 0 ? 1.0 : 0.0 for n in occupations]
    else
        weights = exp.(-ω .* occupations ./ kBT)
        weights ./= sum(weights)
    end
    number_states = [MPS([physical_site], [string(n)]) for n in occupations]
    density = to_dm(number_states; coeffs=weights)
    return to_liouville(density; sites=[liouville_site])
end

function lorentzian_spinboson_bath(
    kBT;
    n_modes::Integer=SB_N_MODES,
    local_dim::Integer=SB_LOCAL_DIM,
    C::Real=SB_C,
)
    frequencies, couplings = lorentzian_mode_grid(n_modes; C)
    N_modes = length(frequencies)
    bath_sites = siteinds("Boson", N_modes; dim=local_dim)
    bath_liouville_sites = liouv_sites(bath_sites)
    modes = BosonicMode[]
    for k in 1:N_modes
        ωk = frequencies[k]
        gk = couplings[k]
        mode_hamiltonian = OpSum()
        mode_hamiltonian += ωk, "N", 1
        mode_coupling = OpSum()
        mode_coupling += gk, "A", 1, "ProjUp", 2
        mode_coupling += gk, "Adag", 1, "ProjUp", 2
        if abs(ωk) > 1e-12
            mode_coupling += gk^2 / ωk, "ProjUp", 2
        end
        push!(
            modes,
            bosonic_mode(
                [bath_liouville_sites[k]],
                mode_hamiltonian,
                thermal_boson_density(
                    bath_sites[k],
                    bath_liouville_sites[k],
                    ωk,
                    kBT,
                    local_dim,
                );
                coupling=mode_coupling,
            ),
        )
    end
    return with_logger(NullLogger()) do
        bosonic_bath(modes)
    end
end

function driven_tls_system()
    sites = siteinds("S=1/2", 1)
    H = OpSum()
    # ITensor Sx = σx/2, so Ω Sx = (Ω/2) σx = ħΩ/2 σx with ħ = Ω = 1.
    H += SB_OMEGA, "Sx", 1
    return with_logger(NullLogger()) do
        spin_system(sites, H)
    end
end

function build_ace_pt(system, bath; dt, nsteps, cutoff, maxdim, compression=:zipup_cpp)
    return with_logger(NullLogger()) do
        build_process_tensor(
            system;
            method=ACE(cutoff=cutoff, maxdim=maxdim, compression=compression),
            environment=bath,
            dt,
            nsteps,
            alg=Exact(),
            sys_alg=Trotter{2}(),
            combine_alg=Trotter{2}(),
            progress=false,
            verbose=false,
        )
    end
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

"""Untimed compile/warmup build, then BenchmarkTools samples of construction only."""
function measure_ace_pt(system, bath; dt, nsteps, cutoff, maxdim, compression=:zipup_cpp)
    pt = build_ace_pt(system, bath; dt, nsteps, cutoff, maxdim, compression)
    χ = Int(maxlinkdim(pt.core))
    bench = @benchmarkable build_ace_pt(
        $system,
        $bath;
        dt=$dt,
        nsteps=$nsteps,
        cutoff=$cutoff,
        maxdim=$maxdim,
        compression=$compression,
    )
    trial = run(bench; samples=BENCH_SAMPLES, evals=1, seconds=BENCH_SECONDS)
    return pt, trial_stats(trial), χ
end
