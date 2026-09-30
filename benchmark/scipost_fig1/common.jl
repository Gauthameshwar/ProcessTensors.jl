# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# Shared model, exact-diagonalization, process-tensor, and data helpers for
# the SciPost Figure 1 accuracy benchmark.
#
# Four-mode Sz⊗Sz spin bath matching scripts/pt_tfim_multimode.jl. The
# observable is Pauli ⟨σy⟩. Snapshot k from evolve is compared with ED at
# t = k Δt; t = 0 is the unevolved product initial state.

include(joinpath(@__DIR__, "..", "env.jl"))

using Dates
using ITensors
using ITensors.Ops: Exact, Trotter
using LinearAlgebra
using ProcessTensors
using Printf

const RESULTS_DIR = joinpath(@__DIR__, "results")

const NMODES = 4
const OMEGA_SYSTEM = 1.0
const MODE_FREQUENCIES = [0.5 + 0.1 * m for m in 1:NMODES]
const MODE_COUPLINGS = [0.2 + 0.3 * m for m in 1:NMODES]
const MODE_AXES = ntuple(_ -> "Sz", NMODES)
const DEFAULT_FINAL_TIME = 6.0
const DEFAULT_ACE_CUTOFF = 1e-12
const PAULI_Y = ComplexF64[0 -im; im 0]
const DSYS = 2
const DENV = 2^NMODES

"""Contract a Hilbert-space MPO into a dense matrix in `physical_sites` order."""
function hilbert_mpo_to_dense(ρ::AbstractMPO{Hilbert}, physical_sites)
    tensor = foldl(*, ρ.core)
    array = Array(tensor, prime.(physical_sites)..., physical_sites...)
    dimension = prod(dim.(physical_sites))
    return reshape(ComplexF64.(array), dimension, dimension)
end

dense_hamiltonian_matrix(H::OpSum, physical_sites) =
    hilbert_mpo_to_dense(MPO(H, physical_sites), physical_sites)

"""
Trace out the bath from a joint density whose ITensor site order is
`[system, bath_1, …]`, matching `scripts/pt_tfim_multimode.jl`.
"""
function partial_trace_system(ρ::AbstractMatrix{<:Number}, dsys::Int, denv::Int)
    ρ4 = reshape(ComplexF64.(ρ), dsys, denv, dsys, denv)
    reduced = zeros(ComplexF64, dsys, dsys)
    for env_index in 1:denv
        reduced .+= @view ρ4[:, env_index, :, env_index]
    end
    return reduced
end

function one_site_density_matrix(ρ::AbstractMPO{Hilbert})
    tensor = foldl(*, ρ.core)
    site = only(filter(index -> plev(index) == 0 && hastags(index, "Site"), inds(tensor)))
    return ComplexF64.(Array(tensor, prime(site), site))
end

sigma_y(ρ::AbstractMatrix{<:Number}) = real(tr(ρ * PAULI_Y))

function parse_float_list(value::AbstractString)
    values = parse.(Float64, strip.(split(value, ',')))
    isempty(values) && throw(ArgumentError("Expected at least one floating-point value."))
    return values
end

"""Number of process-tensor slabs so that the last snapshot is compared at `T`."""
function nsteps_for(final_time::Real, dt::Real)
    dt > 0 || throw(ArgumentError("dt must be positive; got $dt."))
    final_time > 0 || throw(ArgumentError("final_time must be positive; got $final_time."))
    steps = round(Int, final_time / dt)
    isapprox(steps * dt, final_time; atol=100eps(Float64) * max(1, abs(final_time))) ||
        throw(ArgumentError("final_time=$final_time must be an integer multiple of dt=$dt."))
    steps >= 1 || throw(ArgumentError("Need at least one timestep; got nsteps=$steps."))
    return steps
end

"""
Four-mode Sz⊗Sz spin bath matching `scripts/pt_tfim_multimode.jl`.

All modes couple through Sz on the system, so ACE `combine_alg` (M1 vs M2)
agrees to numerical precision.
"""
function multimode_spin_model()
    sys_phys = siteinds("S=1/2", 1)
    env_phys = siteinds("S=1/2", NMODES)
    env_liouv = liouv_sites(env_phys)

    H_sys = OpSum() + (OMEGA_SYSTEM, "Sx", 1)
    system = spin_system(sys_phys, H_sys)
    ρ_sys0 = to_dm(MPS(sys_phys, ["Up"]))

    modes = SpinMode[]
    for m in 1:NMODES
        ρ_mode = to_liouville(to_dm(MPS([env_phys[m]], ["Up"])); sites=[env_liouv[m]])
        H_mode = OpSum() + (MODE_FREQUENCIES[m], "Sx", 1)
        coupling = OpSum() + (MODE_COUPLINGS[m], "Sz", 1, "Sz", 2)
        push!(modes, spin_mode([env_liouv[m]], H_mode, ρ_mode; coupling=coupling))
    end
    bath = spin_bath(modes)

    joint_sites = Index[sys_phys[1], env_phys...]
    H_full = OpSum() + (OMEGA_SYSTEM, "Sx", 1)
    for m in 1:NMODES
        H_full += MODE_FREQUENCIES[m], "Sx", m + 1
        H_full += MODE_COUPLINGS[m], "Sz", m + 1, "Sz", 1
    end
    ρ_joint0 = hilbert_mpo_to_dense(
        to_dm(MPS(joint_sites, vcat(["Up"], fill("Up", NMODES)))),
        joint_sites,
    )

    return (; system, bath, ρ_sys0, sys_phys, joint_sites, H_full, ρ_joint0)
end

"""
Discretisation-free Hilbert-space ED of the joint 32-dimensional Hamiltonian.

Times are compared with Pauli ⟨σy⟩ after tracing out the four bath spins.
"""
function direct_ed(model, times)
    H = Hermitian(dense_hamiltonian_matrix(model.H_full, model.joint_sites))
    decomposition = eigen(H)
    values = Float64[]
    for t in times
        if iszero(t)
            ρ_system = partial_trace_system(model.ρ_joint0, DSYS, DENV)
        else
            phases = exp.(-1im * float(t) .* decomposition.values)
            U = decomposition.vectors * Diagonal(phases) * decomposition.vectors'
            ρ_joint = U * model.ρ_joint0 * U'
            ρ_system = partial_trace_system(ρ_joint, DSYS, DENV)
        end
        push!(values, sigma_y(ρ_system))
    end
    return values
end

function process_tensor_sigma_y(
    model;
    method,
    dt::Real,
    nsteps::Int,
    sys_alg,
    combine_alg=Trotter{1}(),
)
    started = time_ns()
    pt = build_process_tensor(
        model.system, model.system.sites[1];
        method=method,
        environment=model.bath,
        dt=dt,
        nsteps=nsteps,
        alg=Exact(),
        sys_alg=sys_alg,
        combine_alg=combine_alg,
        progress=false,
    )
    build_seconds = (time_ns() - started) / 1e9
    trajectory = evolve(pt, model.ρ_sys0; progress=false)
    length(trajectory.states_hilbert) == nsteps || throw(
        ArgumentError(
            "evolve returned $(length(trajectory.states_hilbert)) snapshots; expected $nsteps.",
        ),
    )

    # t = 0 is the unevolved initial state. Snapshot k is compared at t = k Δt,
    # matching scripts/pt_tfim_multimode.jl (not evolve's labelled times).
    times = collect(range(0.0; step=dt, length=nsteps + 1))
    sy = Vector{Float64}(undef, nsteps + 1)
    sy[1] = sigma_y(one_site_density_matrix(model.ρ_sys0))
    for k in 1:nsteps
        sy[k + 1] = sigma_y(one_site_density_matrix(trajectory.states_hilbert[k]))
    end
    return (; times, sy, chi_max=maxlinkdim(pt.core), build_seconds)
end

function exact_pt_trajectory(model; dt::Real, nsteps::Int, sys_order::Int)
    sys_alg = sys_order == 1 ? Trotter{1}() :
              sys_order == 2 ? Trotter{2}() :
              throw(ArgumentError("sys_order must be 1 or 2; got $sys_order."))
    return process_tensor_sigma_y(model; method=Dense(), dt, nsteps, sys_alg)
end

function ace_trajectory(
    model;
    dt::Real,
    nsteps::Int,
    sys_order::Int,
    mode_order::Int,
    cutoff::Real=DEFAULT_ACE_CUTOFF,
)
    sys_alg = sys_order == 1 ? Trotter{1}() :
              sys_order == 2 ? Trotter{2}() :
              throw(ArgumentError("sys_order must be 1 or 2; got $sys_order."))
    combine_alg = mode_order == 1 ? Trotter{1}() :
                  mode_order == 2 ? Trotter{2}() :
                  throw(ArgumentError("mode_order must be 1 or 2; got $mode_order."))
    method = ACE(cutoff=cutoff, compression=:canonzip)
    return process_tensor_sigma_y(
        model; method, dt, nsteps, sys_alg, combine_alg,
    )
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

function write_environment()
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
        println(io, "OPENBLAS_NUM_THREADS = ", get(ENV, "OPENBLAS_NUM_THREADS", "unset"))
        println(io, "model = four-mode Sz⊗Sz couplings, T=$DEFAULT_FINAL_TIME")
        println(io, "observable = Pauli sigma_y")
    end
    println("Wrote $path")
    return path
end
