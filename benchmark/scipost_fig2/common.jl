# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# Shared model, reproducibility, bond-profile, and CSV helpers for SciPost
# Figure 2.

using Dates
using ITensors
using ITensors.Ops: Exact, Trotter
using LinearAlgebra
using Logging
using ProcessTensors
using Printf
using Random
using SHA

const RESULTS_DIR = joinpath(@__DIR__, "results")
const ORIENTATIONS_PATH = joinpath(RESULTS_DIR, "bath_orientations.csv")

const DEFAULT_SEED = 20260905
const DEFAULT_N_BATH = 50
const DEFAULT_J = 1.0
# T must be an integer multiple of every Δt, including 1.5, so T = 21
# rather than 20.
const DEFAULT_FINAL_TIME = 21.0
const DEFAULT_DT_REFERENCE = 0.10
const DEFAULT_TIMESTEPS = (1.5, 0.5, 0.20, 0.10, 0.05)
const DEFAULT_CUTOFFS = (1e-6, 1e-8, 1e-10, 1e-12)
const DEFAULT_MAXDIM = 4096

function parse_float_list(value::AbstractString)
    values = parse.(Float64, strip.(split(value, ',')))
    isempty(values) && throw(ArgumentError("Expected at least one floating-point value."))
    return values
end

function nsteps_for(final_time::Real, dt::Real)
    dt > 0 || throw(ArgumentError("dt must be positive; got $dt."))
    final_time > 0 || throw(ArgumentError("final_time must be positive; got $final_time."))
    steps = round(Int, final_time / dt)
    isapprox(steps * dt, final_time; atol=100eps(Float64) * max(1, abs(final_time))) ||
        throw(ArgumentError("final_time=$final_time must be an integer multiple of dt=$dt."))
    steps >= 2 || throw(ArgumentError("Need at least two timesteps to define an internal bond."))
    return steps
end

"""
Draw `n_bath` pure-spin orientations uniformly on the Bloch sphere.

The local `Xoshiro` stream makes the realization independent of the global RNG.
"""
function sample_bath_orientations(n_bath::Int, seed::Integer)
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

function read_orientations(path::AbstractString)
    lines = readlines(path)
    isempty(lines) && error("Empty orientation file: $path")
    strip(first(lines)) == "k,theta,phi,nx,ny,nz" ||
        error("Unexpected orientation header in $path")
    orientations = NamedTuple[]
    for line in Iterators.drop(lines, 1)
        isempty(strip(line)) && continue
        fields = split(line, ',')
        length(fields) == 6 || error("Malformed orientation row: $line")
        push!(
            orientations,
            (
                k=parse(Int, fields[1]),
                theta=parse(Float64, fields[2]),
                phi=parse(Float64, fields[3]),
                nx=parse(Float64, fields[4]),
                ny=parse(Float64, fields[5]),
                nz=parse(Float64, fields[6]),
            ),
        )
    end
    return orientations
end

orientation_hash(path::AbstractString) = bytes2hex(sha256(read(path)))

function bloch_spin_density(physical_site, liouville_site, orientation)
    θ = orientation.theta
    ϕ = orientation.phi
    amplitudes = ComplexF64[cos(θ / 2), cis(ϕ) * sin(θ / 2)]
    ket = MPS(ITensor(amplitudes, physical_site), [physical_site])
    return to_liouville(to_dm(ket); sites=[liouville_site])
end

"""
Construct Cygorek's unpolarized (`b=0`) central-spin bath for one fixed
realization of pure bath-spin states.
"""
function unpolarized_central_spin_bath(orientations; J::Real=DEFAULT_J)
    n_bath = length(orientations)
    n_bath >= 1 || throw(ArgumentError("Need at least one bath orientation."))
    bath_sites = siteinds("S=1/2", n_bath)
    bath_liouville_sites = liouv_sites(bath_sites)
    coupling_strength = J / n_bath
    modes = SpinMode[]
    with_logger(NullLogger()) do
        for k in 1:n_bath
            coupling = OpSum()
            coupling += coupling_strength, "Sx", 1, "Sx", 2
            coupling += coupling_strength, "Sy", 1, "Sy", 2
            coupling += coupling_strength, "Sz", 1, "Sz", 2
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
    initial_density = to_dm(MPS(sites, ["+"]))
    return (; system, initial_density, sites)
end

function build_ace_bond_profile(
    system,
    bath;
    dt::Real,
    nsteps::Int,
    cutoff::Real,
    maxdim::Int,
)
    started = time_ns()
    pt = with_logger(NullLogger()) do
        build_process_tensor(
            system;
            method=ACE(cutoff=cutoff, maxdim=maxdim, compression=:canonzip),
            environment=bath,
            dt,
            nsteps,
            alg=Exact(),
            sys_alg=Trotter{2}(),
            combine_alg=Trotter{2}(),
            progress=false,
        )
    end
    build_seconds = (time_ns() - started) / 1e9
    internal_dims = Int.(linkdims(pt.core))
    length(internal_dims) == nsteps - 1 || error(
        "Expected $(nsteps - 1) internal temporal bonds, got $(length(internal_dims)).",
    )
    dmax = maximum(internal_dims)
    hit_maxdim = any(==(maxdim), internal_dims)

    # Include the two physical MPO boundaries, both exactly one-dimensional.
    times = collect(range(0.0; step=dt, length=nsteps + 1))
    dimensions = vcat(1, internal_dims, 1)
    return (; times, dimensions, dmax, hit_maxdim, build_seconds)
end

function write_csv(path::AbstractString, header, rows)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(header, ','))
        for row in rows
            println(io, join(row, ','))
        end
    end
    println("Wrote $path")
    return path
end

function module_version(mod::Module)
    try
        return string(pkgversion(mod))
    catch
        return "unknown"
    end
end

function write_environment(
    ;
    seed,
    n_bath,
    J,
    final_time,
    dt_reference,
    timesteps,
    cutoffs,
    maxdim,
    orientations_path,
)
    mkpath(RESULTS_DIR)
    path = joinpath(RESULTS_DIR, "environment.txt")
    open(path, "w") do io
        println(io, "generated_utc = ", now(UTC))
        println(io, "cpu = AMD Ryzen Threadripper 7980X (64 cores, 128 threads)")
        println(io, "ram = 251 GiB")
        println(io, "os = Ubuntu 22.04.5 LTS x86_64")
        println(io, "julia = ", VERSION)
        println(io, "julia_threads = ", Threads.nthreads())
        println(io, "blas_threads = ", BLAS.get_num_threads())
        println(io, "blas = ", BLAS.get_config())
        println(io, "ProcessTensors = ", module_version(ProcessTensors))
        println(io, "ITensors = ", module_version(ITensors))
        println(io, "rng = Xoshiro")
        println(io, "seed = ", seed)
        println(io, "N_bath = ", n_bath)
        println(io, "J = ", J)
        println(io, "T = ", final_time)
        println(io, "dt_reference = ", dt_reference)
        println(io, "timesteps = ", join(timesteps, ','))
        println(io, "cutoffs = ", join(cutoffs, ','))
        println(io, "ACE_compression = canonzip")
        println(io, "ACE_maxdim = ", maxdim)
        println(io, "orientation_file = ", basename(orientations_path))
        println(io, "orientation_sha256 = ", orientation_hash(orientations_path))
    end
    println("Wrote $path")
    return path
end
