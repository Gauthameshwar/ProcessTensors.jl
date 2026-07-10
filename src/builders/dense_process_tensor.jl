# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: src/builders/dense_process_tensor.jl
# Contributor: Gauthameshwar S.
#
# Implements dense process-tensor core construction for no-bath, single-mode,
# and small multimode environments.

import ITensors: terms
import ITensors.Ops: Exact, Trotter

const MAX_DENSE_LIOUVILLE_DIM = 5_000

function _validate_dense_liouville_budget(d_joint::Integer; context::AbstractString)
    d_joint <= MAX_DENSE_LIOUVILLE_DIM && return nothing
    @warn "$context: joint Liouville vector dimension D=$d_joint exceeds MAX_DENSE_LIOUVILLE_DIM=$(MAX_DENSE_LIOUVILLE_DIM)."
    throw(
        ArgumentError(
            "$context: joint Liouville vector dimension D=$d_joint is too large for dense exp(dt * L). " *
            "Please reduce mode count / local cutoff or wait for TEBD-based large-bath support.",
        ),
    )
end

# Internal prime level for the system propagator fused into a core.
const _INTERNAL_PLEV = 2

# One-step embedded system Liouville map on PT legs `(in_k, out_k)`.
function _system_liouvillian_pt_core(
    system::AbstractSystem,
    in_k::Index,
    out_k::Index,
    dt::Real;
    alg=Trotter{2}(),
)
    U = liouvillian_propagator_itensor(
        system.H,
        system.sites,
        dt;
        alg=alg,
        jump_ops=system.jump_ops,
    )
    gate_site = only(system.sites)
    return replaceind(replaceind(U, prime(gate_site), in_k), gate_site, out_k)
end

# Build Markovian PT cores with embedded system propagation on each `(in_k, out_k)` pair.
function _build_trivial_pt_cores(
    system::AbstractSystem,
    coupling_site::Index,
    dt::Real,
    nsteps::Int;
    alg=Trotter{2}(),
)
    cores = ITensor[]
    for k in 0:(nsteps - 1)
        in_k, out_k = generate_pt_legs(coupling_site, k)
        push!(cores, _system_liouvillian_pt_core(system, in_k, out_k, dt; alg=alg))
    end
    return cores
end

joint_liouville_dim(bath::AbstractBath, coupling_site::Index) =
    prod(dim.(collect(Index[vcat([only(m.sites) for m in bath.modes], [coupling_site])...])))

# Build one PT core per timestep by embedding a joint bath-system propagator and retaining one bath memory link.
function _build_bathmode_pt_cores(
    system::AbstractSystem,
    coupling_site::Index,
    bathmode::AbstractBathMode,
    dt::Real,
    nsteps::Int;
    bath_coupling::OpSum=OpSum(),
    alg=Exact(),
    sys_alg=Trotter{2}(),
    kwargs...
)
    length(bathmode.sites) == 1 || throw(
        ArgumentError("build_process_tensor: AbstractBathMode must have exactly one site index. Got $(length(bathmode.sites)).")
    )
    env_liouv = only(bathmode.sites)
    d_env = dim(env_liouv)
    d_sys = dim(coupling_site)
    d_joint = d_env * d_sys
    _validate_dense_liouville_budget(d_joint; context="_build_bathmode_pt_cores")

    coupling_term = bathmode.coupling == OpSum() ? bath_coupling : bathmode.coupling
    # Joint physical Hamiltonian on [bath, system]; mode coupling uses sites 1=bath, 2=system.
    joint_ops = bathmode.H + coupling_term
    sites_vec = Index[env_liouv, coupling_site]
    U_ref = liouvillian_propagator_itensor(joint_ops, sites_vec, dt; alg=alg)

    # Bath virtual memory legs: nsteps cores use nsteps+1 links.
    bath_links = [Index(d_env; tags="PT,Link,tstep=$k") for k in 0:nsteps]

    cores = ITensor[]
    inputs = Index[]
    outputs = Index[]
    for k in 0:(nsteps - 1)
        in_k, out_k = generate_pt_legs(coupling_site, k)
        push!(inputs, in_k)
        push!(outputs, out_k)
        left = bath_links[k + 1]
        right = bath_links[k + 2]

        core_k = replaceind(U_ref, prime(env_liouv), right)
        core_k = replaceind(core_k, env_liouv, left)
        core_k = replaceind(core_k, prime(coupling_site), in_k)
        core_k = replaceind(core_k, coupling_site, out_k)
        push!(cores, core_k)
    end
    for k in 0:(nsteps - 1)
        in_k = inputs[k + 1]
        out_k = outputs[k + 1]
        internal_out = prime(out_k, _INTERNAL_PLEV)
        core_k = replaceind(cores[k + 1], out_k, internal_out)
        sys_prop = replaceind(
            _system_liouvillian_pt_core(system, in_k, out_k, dt; alg=sys_alg),
            in_k,
            internal_out,
        )
        cores[k + 1] = core_k * sys_prop
    end
    # Contract the first and last bath links with the initial bath state and the trace out
    initial_bath_state = instrument_itensor(StatePreparation(bathmode.rho0), [bath_links[1]'], 0)
    noprime!(initial_bath_state)
    trace_out = instrument_itensor(TraceOut(), [bath_links[end]], nsteps)

    cores[1] *= initial_bath_state
    cores[end] *= trace_out

    return cores
end

# Build PT cores for multiple modes using a fused bath memory link between timesteps.
function _build_multimode_pt_cores(
    system::AbstractSystem,
    coupling_site::Index,
    environment::AbstractBath,
    dt::Real,
    nsteps::Int;
    alg=Exact(),
    sys_alg=Trotter{2}(),
    kwargs...
)
    isempty(environment.modes) && throw(ArgumentError("_build_multimode_pt_cores: environment must contain at least one mode."))

    modes = environment.modes
    nmodes = length(modes)
    sys_site = nmodes + 1
    sites_vec = Index[vcat([only(m.sites) for m in modes], [coupling_site])...]
    d_joint = prod(dim.(collect(sites_vec)))
    _validate_dense_liouville_budget(d_joint; context="_build_multimode_pt_cores")

    joint_ops = OpSum()
    for (i, mode) in enumerate(modes)
        for term in ITensors.terms(mode.H)
            c = ITensors.coefficient(term)
            args = Any[]
            for op_t in collect(last(term.args))
                push!(args, ITensors.name(op_t))
                for s in ITensors.sites(op_t)
                    src = Int(s)
                    src == 1 || throw(ArgumentError("_build_multimode_pt_cores(mode.H): expected local site 1, got $src."))
                    push!(args, i)
                end
            end
            joint_ops += (c, args...)
        end
        for term in ITensors.terms(mode.coupling)
            c = ITensors.coefficient(term)
            args = Any[]
            for op_t in collect(last(term.args))
                push!(args, ITensors.name(op_t))
                for s in ITensors.sites(op_t)
                    src = Int(s)
                    dst = if src == 1
                        i
                    elseif src == 2
                        sys_site
                    else
                        throw(ArgumentError("_build_multimode_pt_cores(mode.coupling): expected local sites 1 or 2, got $src."))
                    end
                    push!(args, dst)
                end
            end
            joint_ops += (c, args...)
        end
    end
    joint_ops += environment.coupling

    U_ref = liouvillian_propagator_itensor(joint_ops, sites_vec, dt; alg=alg)

    bath_sites = collect(sites_vec[1:(end - 1)])
    bath_sites_prime = prime.(bath_sites)
    comb_unprimed = combiner(bath_sites...; tags="PT,Link,FusedBath")
    comb_primed = combiner(bath_sites_prime...; tags="PT,Link,FusedBath,Prime")
    U_ref = U_ref * comb_unprimed * comb_primed

    fused_left = combinedind(comb_unprimed)
    fused_right = combinedind(comb_primed)
    d_bath = prod(dim.(bath_sites))
    bath_links = [Index(d_bath; tags="PT,Link,tstep=$k") for k in 0:nsteps]

    cores = ITensor[]
    inputs = Index[]
    outputs = Index[]
    for k in 0:(nsteps - 1)
        in_k, out_k = generate_pt_legs(coupling_site, k)
        push!(inputs, in_k)
        push!(outputs, out_k)
        left = bath_links[k + 1]
        right = bath_links[k + 2]

        core_k = replaceind(U_ref, prime(coupling_site), in_k)
        core_k = replaceind(core_k, coupling_site, out_k)
        core_k = replaceind(core_k, fused_left, left)
        core_k = replaceind(core_k, fused_right, right)
        push!(cores, core_k)
    end
    for k in 0:(nsteps - 1)
        in_k = inputs[k + 1]
        out_k = outputs[k + 1]
        internal_out = prime(out_k, _INTERNAL_PLEV)
        core_k = replaceind(cores[k + 1], out_k, internal_out)
        sys_prop = replaceind(
            _system_liouvillian_pt_core(system, in_k, out_k, dt; alg=sys_alg),
            in_k,
            internal_out,
        )
        cores[k + 1] = core_k * sys_prop
    end

    bath_state = ITensor(1.0)
    for mode in modes
        site = only(mode.sites)
        prep = instrument_itensor(StatePreparation(mode.rho0), Index[prime(site)], 0)
        noprime!(prep)
        hasind(prep, site) || throw(ArgumentError("_build_multimode_pt_cores: prepared mode state is missing mode site index."))
        bath_state *= prep
    end
    initial_bath_state = replaceind(bath_state * comb_unprimed, combinedind(comb_unprimed), bath_links[1])

    bath_trace = ITensor(1.0)
    for site in bath_sites_prime
        bath_trace *= Instruments._vectorized_identity_itensor(Index[site])
    end
    trace_out = replaceind(bath_trace * comb_primed, combinedind(comb_primed), bath_links[end])

    cores[1] *= initial_bath_state
    cores[end] *= trace_out

    return cores
end
