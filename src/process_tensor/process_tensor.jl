# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: src/process_tensor/process_tensor.jl
# Contributor: Gauthameshwar S.
#
# Defines the ProcessTensor wrapper, time-leg helpers, default schedules, and
# schedule validation utilities.

import ITensorMPS: MPO as CoreMPO, linkdims, maxlinkdim
import ITensors: terms
import Base: getproperty, setproperty!, show

"""
    ProcessTensor

Liouville-space MPO wrapper for a single-coupling-site process tensor.

Stores the environment influence and one-step system Liouvillian propagation as
an `MPO{Liouville}` core over time-ordered input/output legs.
Fields `system`, `environment`, `dt`, `nsteps`, and `coupling_site` record the
physical model. High-level APIs such as [`evaluate_process`](@ref),
[`evolve`](@ref), and [`two_time_correlation_seq`](@ref) contract this object
with instrument schedules.

Unknown property access delegates to `.core`, like [`MPO`](@ref).

# Examples
```julia
pt = build_process_tensor(system, coupling_site; dt=0.1, nsteps=8)
trajectory = evolve(pt, ρ0)
```
"""
struct ProcessTensor{S<:AbstractSystem,E} <: AbstractMPO{Liouville}
    core::CoreMPO
    system::S
    environment::E
    dt::Float64
    nsteps::Int
    coupling_site::Index

    function ProcessTensor(
        core::CoreMPO,
        system::S,
        environment::E,
        dt::Real,
        nsteps::Integer,
        coupling_site::Index,
    ) where {S<:AbstractSystem,E<:Union{Nothing,AbstractBath}}
        nsteps_int = Int(nsteps)
        nsteps_int >= 1 || throw(ArgumentError("A process tensor requires at least one timestep; got $nsteps."))
        _validate_coupling_site(system, coupling_site)
        length(core) == nsteps_int || throw(
            ArgumentError("ProcessTensor core length must equal nsteps for single-site PT. Got length(core)=$(length(core)) and nsteps=$nsteps_int."),
        )
        return new{S,E}(core, system, environment, float(dt), nsteps_int, coupling_site)
    end
end

function _validate_coupling_site(system::AbstractSystem, coupling_site::Index)
    isempty(system.sites) && throw(ArgumentError("System sites cannot be empty."))
    coupling_site in system.sites || throw(
        ArgumentError("coupling_site must be one of system.sites in Liouville space."),
    )
    return nothing
end

function ProcessTensor(
    core::CoreMPO,
    system::AbstractSystem,
    environment,
    dt::Real,
    nsteps::Integer,
)
    length(system.sites) == 1 || throw(
        ArgumentError("ProcessTensor(core, system, environment, dt, nsteps) is only allowed for single-site systems. Pass coupling_site::Index explicitly."),
    )
    return ProcessTensor(core, system, environment, dt, nsteps, only(system.sites))
end

function Base.getproperty(pt::ProcessTensor, sym::Symbol)
    sym in fieldnames(typeof(pt)) && return getfield(pt, sym)
    return getproperty(getfield(pt, :core), sym)
end

function Base.setproperty!(pt::ProcessTensor, sym::Symbol, val)
    sym in fieldnames(typeof(pt)) && return setfield!(pt, sym, val)
    return setproperty!(getfield(pt, :core), sym, val)
end


function _environment_summary(environment)
    if environment === nothing
        return "none"
    elseif environment isa AbstractBath
        env = environment
        nmodes = length(env.modes)
        d_bath = isempty(env.sites) ? 1 : prod(dim.(env.sites))
        has_coupling = !isempty(terms(env.coupling)) ||
            any(!isempty(terms(m.coupling)) for m in env.modes)
        return string(
            nameof(typeof(env)),
            "(nmodes=", nmodes, ", D_bath=", d_bath, ", coupling=", has_coupling, ")",
        )
    else
        return string(typeof(environment))
    end
end

struct _InfoText
    text::String
end

Base.show(io::IO, item::_InfoText) = print(io, item.text)

_info_text(value::AbstractString) = _InfoText(value)

function Base.show(io::IO, pt::ProcessTensor)
    S, E = typeof(pt).parameters
    t_final = pt.dt * pt.nsteps
    χ = maxlinkdim(pt.core)
    print(io, pt.nsteps, "-step ProcessTensor{", nameof(S), ", ", nameof(E), "}")
    print(io, " | dt=", pt.dt, " | t_final=", round(t_final, digits=10), " | maxlinkdim=", χ)
    println(io)
    println(io)
    nsites = length(pt.system.sites)
    println(io, "  system:      ", nameof(S), "(nsites=", nsites, ", dissipative=", !isempty(pt.system.jump_ops), ")")
    print(io, "  environment: ")
    println(io, _environment_summary(pt.environment))
    ldims = Int[d for d in linkdims(pt.core) if d !== nothing]
    print(io, "  core:        MPO{Liouville}(length=", length(pt.core), ", linkdims=")
    if isempty(ldims)
        print(io, "none")
    else
        print(io, "[")
        if length(ldims) <= 10
            print(io, join(ldims, ", "))
        else
            print(io, join(ldims[1:3], ", "), ", …, ", ldims[end])
        end
        print(io, "]")
    end
    println(io, ")")
end

Base.show(io::IO, ::MIME"text/plain", pt::ProcessTensor) = show(io, pt)

function _tagset_with_tstep(s::Index, k::Int)
    tokens = filter(token -> token != "Site" && !startswith(token, "tstep="), tag_tokens(s))
    return ITensors.TagSet(join(vcat(tokens, ["tstep=$k"]), ","))
end

"""
    generate_pt_legs(site::Index, k::Int)

Construct the input/output Liouville legs for one process-tensor timestep.

The returned tuple is `(input_site, output_site)`. The input leg is the primed
version (`plev = 1`) of the output leg (`plev = 0`), and both carry the same
physical Liouville metadata as `site` plus a `tstep=k` tag.
"""
function generate_pt_legs(site::Index, k::Int)
    output_site = Index(dim(site); tags=_tagset_with_tstep(site, k))
    return prime(output_site), output_site
end


_schedule_default_instr(::ProcessTensor) = IdentityOperation()

# Thin PT wrapper around the seq-first canonical coverage API.
Instruments.instrument_leg_maps(pt::ProcessTensor, seq::InstrumentSeq) =
    instrument_leg_maps(seq, pt.nsteps)

# Shared schedule validation for the lazy evaluation pipeline.
function _validate_instrument_schedule!(
    pt::ProcessTensor,
    seq::InstrumentSeq,
    default_instr::AbstractInstrument,
    caller::AbstractString,
)
    _, _, missing_in, missing_out = instrument_leg_maps(seq, pt.nsteps)
    isempty(missing_in) || throw(
        ArgumentError("$caller: missing input legs for tsteps $(missing_in)."),
    )
    isempty(missing_out) || throw(
        ArgumentError("$caller: missing output legs for tsteps $(missing_out)."),
    )
    resolve_instrument(seq, 0) isa SingleLegInstrument || throw(
        ArgumentError("$caller: tstep=0 must be a single-leg initial preparation."),
    )
    return nothing
end

"""
    default_schedule(pt)

Construct an [`InstrumentSeq`](@ref) whose ordinary evolve slots use
[`IdentityOperation`](@ref).

The returned schedule does not include an initial state or final trace-out.
Add those explicitly, or use [`evolve`](@ref) / [`evaluate_process`](@ref)
overloads that accept an initial state.
"""
function default_schedule(pt::ProcessTensor)
    return InstrumentSeq(default=_schedule_default_instr(pt), nsteps=pt.nsteps)
end

"""
    output_sites(pt, k)

Return the unprimed output process-tensor leg at time label `k`.
"""
function output_sites(pt::ProcessTensor, k::Int)
    0 <= k < pt.nsteps || throw(BoundsError(0:(pt.nsteps - 1), k))
    core_k = pt.core[k + 1]
    sys_legs = Index[idx for idx in inds(core_k) if !has_tag_token(idx, "Link")]
    length(sys_legs) == 2 || throw(
        ArgumentError(
            "output_sites: core k=$k expected exactly 2 system legs, found $(length(sys_legs)).",
        ),
    )
    candidates = filter(
        idx -> plev(idx) == 0 &&
               tag_value(idx, "tstep=") == string(k) &&
               !has_tag_token(idx, "Link"),
        sys_legs,
    )
    length(candidates) == 1 || throw(
        ArgumentError("output_sites: expected one output leg at tstep=$k, found $(length(candidates))."),
    )
    out = only(candidates)
    plev(out) == 0 || throw(ArgumentError("output_sites: expected plev=0 output leg, got plev=$(plev(out))."))
    return Index[out]
end

"""
    input_sites(pt, k)

Return the primed input process-tensor leg at time label `k`.
"""
function input_sites(pt::ProcessTensor, k::Int)
    0 <= k < pt.nsteps || throw(BoundsError(0:(pt.nsteps - 1), k))
    out = only(output_sites(pt, k))
    inn = prime(out)
    plev(inn) == 1 || throw(ArgumentError("input_sites: expected plev=1 input leg, got plev=$(plev(inn))."))
    inn in inds(pt.core[k + 1]) || throw(
        ArgumentError("input_sites: primed input leg at tstep=$k is not present on core k=$k."),
    )
    return Index[inn]
end

"""
    coupling_times(pt, step)

Return `(out_prev, in_curr)`, the output leg at `step - 1` and input leg at
`step`, used by two-leg instruments between adjacent process-tensor slabs.
"""
function coupling_times(pt::ProcessTensor, step::Int)
    1 <= step <= pt.nsteps || throw(BoundsError(1:pt.nsteps, step))
    if step <= pt.nsteps - 1
        inn = only(input_sites(pt, step))
    else
        # Terminal synthetic input leg (used only for compatibility checks / boundary maps).
        inn = generate_pt_legs(pt.coupling_site, step)[1]
    end
    plev(inn) == 1 || throw(ArgumentError("coupling_times: expected plev=1 on in_curr, got plev=$(plev(inn))."))
    return (output_sites(pt, step - 1), Index[inn])
end

"""
    coupling_sites(pt, step)

Return `(in_curr, out_prev)`, the legacy ordering of [`coupling_times`](@ref).
"""
function coupling_sites(pt::ProcessTensor, step::Int)
    out_prev, in_curr = coupling_times(pt, step)
    return (in_curr, out_prev)
end
