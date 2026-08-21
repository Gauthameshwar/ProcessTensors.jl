# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: benchmark/evolve_contractors/common.jl
# Contributor: Gauthameshwar S.
#
# Shared helpers for evolve contraction benchmarks. Writes CSV output under
# benchmark/evolve_contractors/results/. Runtime and memory come from
# BenchmarkTools after warmup, not a single compiling @elapsed.

import Pkg

const _BENCH_ROOT = dirname(@__DIR__)
const _BENCH_ENV = joinpath(_BENCH_ROOT, ".bench_env")
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
using Logging

const RESULTS_DIR = joinpath(@__DIR__, "results")

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

function trajectory_matrices(pt, rho0)
    trajectory = evolve(pt, rho0; progress=false)
    return [one_site_density_matrix(ρ) for ρ in trajectory.states_hilbert]
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

function empty_spin_system()
    sites = siteinds("S=1/2", 1)
    system = with_logger(NullLogger()) do
        spin_system(sites, OpSum())
    end
    return system, sites
end
