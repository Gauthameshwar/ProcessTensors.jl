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

function _validate_sys_alg(sys_alg)
    (sys_alg isa Trotter{1} || sys_alg isa Trotter{2}) && return nothing
    throw(
        ArgumentError(
            "build_process_tensor: sys_alg must be Trotter{1}() or Trotter{2}() " *
            "(timestep sandwich order of free-system maps around the bath core); " *
            "got $(typeof(sys_alg)). System maps themselves are always Exact (single site).",
        ),
    )
end

# Internal prime level for mid-legs when fusing free-system maps into a bath core.
const _INTERNAL_PLEV = 2

# One-step free-system Liouville map on PT legs `(in_k, out_k)` (always Exact ED).
function _system_liouvillian_pt_core(
    system::AbstractSystem,
    in_k::Index,
    out_k::Index,
    dt::Real,
)
    # Empty free-system generator → identity channel on the PT legs.
    if isempty(ITensors.terms(system.H)) && isempty(system.jump_ops)
        return delta(in_k, out_k)
    end
    gate_site = only(system.sites)
    U = liouvillian_propagator(
        system.H,
        system.sites,
        dt;
        alg=Exact(),
        jump_ops=system.jump_ops,
    )
    # Exact `itensor_exp` uses the ITensor gate convention: unprimed = input,
    # primed = output. Map those onto the PT legs `(in_k, out_k)`.
    return replaceind(replaceind(U, gate_site, in_k), prime(gate_site), out_k)
end

# Asymmetric Trotter{1} sandwich: Q_bath · M_sys(Δt) on the output leg (current default layout).
function _embed_system_map(
    core::ITensor,
    system::AbstractSystem,
    in_k::Index,
    out_k::Index,
    dt::Real,
    ::Trotter{1},
)
    mid = prime(out_k, _INTERNAL_PLEV)
    core_mid = replaceind(core, out_k, mid)
    M = replaceind(_system_liouvillian_pt_core(system, in_k, out_k, dt), in_k, mid)
    return core_mid * M
end

# symmetric Trotter{2} sandwich: M_sys(Δt/2) · Q_bath · M_sys(Δt/2) on input and output legs.
function _embed_system_map(
    core::ITensor,
    system::AbstractSystem,
    in_k::Index,
    out_k::Index,
    dt::Real,
    ::Trotter{2},
)
    mid_in = prime(in_k, _INTERNAL_PLEV)
    mid_out = prime(out_k, _INTERNAL_PLEV)
    half = dt / 2
    # Build half-step maps on temporary PT legs, then retarget the Q-facing leg to `mid_*`.
    M_in = replaceind(_system_liouvillian_pt_core(system, in_k, out_k, half), out_k, mid_in)
    M_out = replaceind(_system_liouvillian_pt_core(system, in_k, out_k, half), in_k, mid_out)
    core_mid = replaceind(replaceind(core, in_k, mid_in), out_k, mid_out)
    return M_in * core_mid * M_out
end

# Build Markovian PT cores with embedded Exact system propagation on each `(in_k, out_k)` pair.
function _build_trivial_pt_cores(
    system::AbstractSystem,
    coupling_site::Index,
    dt::Real,
    nsteps::Int;
    kwargs...,
)
    cores = ITensor[]
    for k in 0:(nsteps - 1)
        in_k, out_k = generate_pt_legs(coupling_site, k)
        push!(cores, _system_liouvillian_pt_core(system, in_k, out_k, dt))
    end
    return cores
end

joint_liouville_dim(bath::AbstractBath, coupling_site::Index) =
    prod(dim.(collect(Index[vcat([only(m.sites) for m in bath.modes], [coupling_site])...])))

# Build one PT core per timestep by embedding a joint bath(+coupling) propagator and retaining one bath memory link.
function _build_bathmode_pt_cores(
    system::AbstractSystem,
    coupling_site::Index,
    bathmode::AbstractBathMode,
    dt::Real,
    nsteps::Int;
    bath_coupling::OpSum=OpSum(),
    alg=Exact(),
    sys_alg=Trotter{1}(),
    kwargs...
)
    _validate_sys_alg(sys_alg)
    length(bathmode.sites) == 1 || throw(
        ArgumentError("build_process_tensor: AbstractBathMode must have exactly one site index. Got $(length(bathmode.sites)).")
    )
    env_liouv = only(bathmode.sites)
    d_env = dim(env_liouv)
    d_sys = dim(coupling_site)
    d_joint = d_env * d_sys
    _validate_dense_liouville_budget(d_joint; context="_build_bathmode_pt_cores")

    coupling_term = bathmode.coupling == OpSum() ? bath_coupling : bathmode.coupling
    # Joint bath(+coupling) slab only; free-system maps are fused via `sys_alg`.
    joint_ops = bathmode.H + coupling_term
    sites_vec = Index[env_liouv, coupling_site]
    U_ref = liouvillian_propagator(joint_ops, sites_vec, dt; alg=alg)

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
        cores[k + 1] = _embed_system_map(
            cores[k + 1], system, inputs[k + 1], outputs[k + 1], dt, sys_alg,
        )
    end
    # Contract the first and last bath links with the initial bath state and the trace out
    initial_bath_state = instrument_itensor(state_preparation(bathmode.rho0), [bath_links[1]'], 0)
    noprime!(initial_bath_state)
    bath_trace = instrument_itensor(trace_out(), [bath_links[end]], nsteps)

    cores[1] *= initial_bath_state
    cores[end] *= bath_trace

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
    sys_alg=Trotter{1}(),
    kwargs...
)
    _validate_sys_alg(sys_alg)
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

    U_ref = liouvillian_propagator(joint_ops, sites_vec, dt; alg=alg)

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
        cores[k + 1] = _embed_system_map(
            cores[k + 1], system, inputs[k + 1], outputs[k + 1], dt, sys_alg,
        )
    end

    bath_state = ITensor(1.0)
    for mode in modes
        site = only(mode.sites)
        prep = instrument_itensor(state_preparation(mode.rho0), Index[prime(site)], 0)
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
