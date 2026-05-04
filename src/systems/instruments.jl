module Instruments

import ..ProcessTensors
import ..ProcessTensors: add!
using ITensors
using ..ProcessTensors: AbstractMPO, AbstractMPS, AbstractSystem, Hilbert, Liouville, MPO,
                        OpSum, OpSum_Liouville, Index, ITensor, dim, plev, prime,
                        replaceind, siteinds, tag_value, to_dm, to_hilbert, to_liouville,
                        _phys_site_from_liouv, _build_trotter_gates

export AbstractInstrument, SingleLegInstrument, TwoLegInstrument,
       StatePreparation, ObservableMeasurement, TraceOut,
       IdentityOperation, SystemPropagation,
       InstrumentSchedule, set_instrument!, instrument_at, resolve_instrument,
       InstrumentSeq, instrument_itensor

abstract type AbstractInstrument end
abstract type SingleLegInstrument <: AbstractInstrument end
abstract type TwoLegInstrument <: AbstractInstrument end

# Prime level convention: input legs are primed (plev=1), output legs are unprimed (plev=0)
const _INPUT_PLEV = 1
const _OUTPUT_PLEV = 0

_assert_valid_leg_plev(leg_plev::Int) =
    leg_plev in (_INPUT_PLEV, _OUTPUT_PLEV) || throw(
        ArgumentError("Instrument leg prime level must be 0 (output) or 1 (input); got $leg_plev."),
    )

function _validate_single_leg_sites(
    instr_name::AbstractString,
    pt_sites::AbstractVector{<:Index},
    leg_plev::Int,
)
    _assert_valid_leg_plev(leg_plev)
    length(pt_sites) == 1 || throw(
        ArgumentError("$instr_name: exactly one process-tensor site is required; got $(length(pt_sites))."),
    )
    all(s -> plev(s) == leg_plev, pt_sites) || throw(
        ArgumentError("$instr_name: sites must all have plev=$leg_plev."),
    )
    return nothing
end

_tstep_from_site(s::Index) = begin
    tstep_str = tag_value(s, "tstep=")
    return tstep_str === nothing ? nothing : parse(Int, tstep_str)
end

function _validate_two_leg_map(
    instr_name::AbstractString,
    input_pt_sites::AbstractVector{<:Index},
    output_pt_sites::AbstractVector{<:Index},
)
    length(input_pt_sites) == 1 || throw(
        ArgumentError("$instr_name: exactly one input process-tensor site is required; got $(length(input_pt_sites))."),
    )
    length(input_pt_sites) == length(output_pt_sites) || throw(
        ArgumentError(
            "$instr_name: requires equal input/output leg counts; got $(length(input_pt_sites)) and $(length(output_pt_sites)).",
        ),
    )
    all(s -> plev(s) == _INPUT_PLEV, input_pt_sites) || throw(
        ArgumentError("$instr_name: input leg sites must have plev=$(_INPUT_PLEV)."),
    )
    all(s -> plev(s) == _OUTPUT_PLEV, output_pt_sites) || throw(
        ArgumentError("$instr_name: output leg sites must have plev=$(_OUTPUT_PLEV)."),
    )
    # Enforce nearest-neighbour coupling in time when both legs carry tstep tags.
    input_tsteps = map(_tstep_from_site, input_pt_sites)
    output_tsteps = map(_tstep_from_site, output_pt_sites)
    for (k, (tin, tout)) in enumerate(zip(input_tsteps, output_tsteps))
        if isnothing(tin) || isnothing(tout)
            continue
        end
        # tout should be one less than tin to connect them via an instrument
        tin == tout + 1 || throw(
            ArgumentError(
                "$instr_name: input site $k has tstep=$tin but output has tstep=$tout (expected $tout + 1 == $tin).",
            ),
        )
    end
    return nothing
end

# =========================================================================
# Single-leg instruments
# =========================================================================

struct StatePreparation{M<:AbstractMPS} <: SingleLegInstrument
    state::M
    pt_sites::Vector{Index}
    leg_plev::Int
end
function StatePreparation(
    state::AbstractMPS,
    pt_sites::AbstractVector{<:Index}=Index[];
    leg_plev::Int=_INPUT_PLEV,
)
    pt_sites_vec = Index[pt_sites...]
    isempty(pt_sites_vec) || _validate_single_leg_sites("StatePreparation", pt_sites_vec, leg_plev)
    return StatePreparation(state, pt_sites_vec, leg_plev)
end

struct ObservableMeasurement{O<:OpSum} <: SingleLegInstrument
    op::O
    pt_sites::Vector{Index}
    leg_plev::Int
end
function ObservableMeasurement(
    op::OpSum,
    pt_sites::AbstractVector{<:Index}=Index[];
    leg_plev::Int=_OUTPUT_PLEV,
)
    pt_sites_vec = Index[pt_sites...]
    isempty(pt_sites_vec) || _validate_single_leg_sites("ObservableMeasurement", pt_sites_vec, leg_plev)
    return ObservableMeasurement(op, pt_sites_vec, leg_plev)
end

struct TraceOut <: SingleLegInstrument
    pt_sites::Vector{Index}
    leg_plev::Int
end
function TraceOut(pt_sites::AbstractVector{<:Index}=Index[]; leg_plev::Int=_OUTPUT_PLEV)
    pt_sites_vec = Index[pt_sites...]
    isempty(pt_sites_vec) || _validate_single_leg_sites("TraceOut", pt_sites_vec, leg_plev)
    return TraceOut(pt_sites_vec, leg_plev)
end

# =========================================================================
# Two-leg instruments
# =========================================================================

struct SystemPropagation{S<:AbstractSystem} <: TwoLegInstrument
    input_pt_sites::Vector{Index}
    output_pt_sites::Vector{Index}
    system::S
end
function SystemPropagation(
    input_pt_sites::AbstractVector{<:Index},
    output_pt_sites::AbstractVector{<:Index},
    system::S,
) where {S<:AbstractSystem}
    input_vec  = Index[input_pt_sites...]
    output_vec = Index[output_pt_sites...]
    _validate_two_leg_map("SystemPropagation", input_vec, output_vec)
    return SystemPropagation{S}(input_vec, output_vec, system)
end
# Lazy constructor used by default schedules; PT leg binding is deferred.
SystemPropagation(system::AbstractSystem) = SystemPropagation(Index[], Index[], system)

struct IdentityOperation <: TwoLegInstrument
    input_pt_sites::Vector{Index}
    output_pt_sites::Vector{Index}
end
function IdentityOperation(
    input_pt_sites::AbstractVector{<:Index},
    output_pt_sites::AbstractVector{<:Index},
)
    input_vec  = Index[input_pt_sites...]
    output_vec = Index[output_pt_sites...]
    _validate_two_leg_map("IdentityOperation", input_vec, output_vec)
    return IdentityOperation(input_vec, output_vec)
end
# Lazy constructor used when explicit PT legs are not required.
IdentityOperation() = IdentityOperation(Index[], Index[])

# =========================================================================
# InstrumentSchedule — compiled schedule (default + per-step overrides)
# =========================================================================

mutable struct InstrumentSchedule
    init::Union{Nothing,StatePreparation}    # Optional initial state prep at tstep=0
    default::AbstractInstrument             # Applied at every unoverridden step
    overrides::Dict{Int,AbstractInstrument} # Per-step overrides for tstep ∈ 1:nsteps
    nsteps::Int                             # Upper bound for validation (0 = unchecked)
end

function InstrumentSchedule(
    default::AbstractInstrument,
    nsteps::Int=0;
    init::Union{Nothing,StatePreparation}=nothing,
    overrides::AbstractDict{Int,<:AbstractInstrument}=Dict{Int,AbstractInstrument}(),
)
    return InstrumentSchedule(
        init, default, Dict{Int,AbstractInstrument}(pairs(overrides)), nsteps,
    )
end

"""
    resolve_instrument(schedule::InstrumentSchedule, k::Int) -> AbstractInstrument or Nothing

Return the instrument scheduled at step `k`.  
`k = 0` returns the `init` field (may be `nothing`).  
`k ≥ 1` returns the override at `k`, or `schedule.default` if none is set.
"""
function resolve_instrument(schedule::InstrumentSchedule, k::Int)
    k == 0 && return schedule.init
    return get(schedule.overrides, k, schedule.default)
end

"""
    set_instrument!(schedule, k, instr) -> schedule

Add or replace the instrument at timestep `k` in a compiled `InstrumentSchedule`.
Use `k = 0` with a `StatePreparation` to set the initial-state override.
"""
function set_instrument!(schedule::InstrumentSchedule, k::Int, instr::AbstractInstrument)
    if k == 0
        instr isa StatePreparation || throw(
            ArgumentError("set_instrument!: Only StatePreparation can be set at tstep=0; got $(typeof(instr))."),
        )
        schedule.init = instr
        return schedule
    end
    k >= 1 || throw(ArgumentError("set_instrument!: Instrument timesteps must be ≥ 0; got $k."))
    if schedule.nsteps > 0 && k > schedule.nsteps
        throw(
            ArgumentError("set_instrument!: tstep=$k exceeds schedule nsteps=$(schedule.nsteps)."),
        )
    end
    schedule.overrides[k] = instr
    return schedule
end

# Backward-compat alias
instrument_at(schedule::InstrumentSchedule, k::Int) = resolve_instrument(schedule, k)

# =========================================================================
# InstrumentSeq — user-facing declarative builder (like OpSum for instruments)
# =========================================================================

"""
    InstrumentSeq

Declarative, order-preserving sequence of `(tstep, instrument)` pairs.

Build a sequence with [`add!`](@ref) or the `+=` shorthand:

```julia
seq = InstrumentSeq()
seq += (StatePreparation(rho0), 0)           # initial state at tstep = 0
seq += (ObservableMeasurement(sz_op), 3)     # measure at tstep = 3
```

Pass to [`evolve`](@ref) for on-the-fly execution, or compile into an
[`InstrumentSchedule`](@ref) with [`create_instruments`](@ref).
"""
struct InstrumentSeq
    entries::Vector{Pair{Int,AbstractInstrument}}
end

InstrumentSeq() = InstrumentSeq(Pair{Int,AbstractInstrument}[])

"""
    add!(seq::InstrumentSeq, instr::AbstractInstrument, tstep::Int) -> seq

Append `instr` at `tstep` to `seq`.  Bound check against a `ProcessTensor`
is deferred to [`create_instruments`](@ref) or [`evolve`](@ref).

Constraints enforced eagerly:
- `tstep ≥ 0`
- Only `StatePreparation` may be placed at `tstep = 0`
"""
function add!(seq::InstrumentSeq, instr::AbstractInstrument, tstep::Int)
    tstep >= 0 || throw(ArgumentError("add!: tstep must be ≥ 0; got $tstep."))
    if tstep == 0 && !(instr isa StatePreparation)
        throw(
            ArgumentError(
                "add!: Only StatePreparation may be placed at tstep=0 (initial condition). Got $(typeof(instr)).",
            ),
        )
    end
    push!(seq.entries, tstep => instr)
    return seq
end

"""
    seq += (instr, tstep)

Mirrors the `OpSum += ("op", site_int)` syntax.
"""
function Base.:+(seq::InstrumentSeq, entry::Tuple{AbstractInstrument,Int})
    add!(seq, entry[1], entry[2])
    return seq
end

function Base.show(io::IO, seq::InstrumentSeq)
    print(io, "InstrumentSeq($(length(seq.entries)) entries)")
    for (k, instr) in seq.entries
        print(io, "\n  tstep=$k => ", typeof(instr))
    end
end

# =========================================================================
# resolve_instrument for InstrumentSeq (on-the-fly path)
# =========================================================================

"""
    resolve_instrument(seq::InstrumentSeq, k::Int, default::AbstractInstrument)
        -> AbstractInstrument or Nothing

Find the instrument for step `k` in `seq` without compiling it first.  
The *last* entry that matches `k` wins (mirrors `create_instruments` semantics).  
Returns `nothing` for `k = 0` when no `StatePreparation` entry exists.
"""
function resolve_instrument(
    seq::InstrumentSeq,
    k::Int,
    default::AbstractInstrument,
)
    found = nothing
    for (tstep, instr) in seq.entries
        tstep == k && (found = instr)
    end
    k == 0 && return found   # nothing if no tstep=0 entry
    return found !== nothing ? found : default
end

# =========================================================================
# instrument_itensor — dense PT-leg tensor builders
# =========================================================================

# Helper functions to convert MPS and MPO to ITensors
function _mps_to_itensor(state::AbstractMPS)
    t = state.core[1]
    for i in 2:length(state.core)
        t *= state.core[i]
    end
    return t
end
function _mpo_to_itensor(op_mpo::AbstractMPO)
    t = op_mpo.core[1]
    for i in 2:length(op_mpo.core)
        t *= op_mpo.core[i]
    end
    return t
end

# Helper function to reindex the ITensor to match the pt_sites
function _reindex_itensor(t::ITensor, old_sites::AbstractVector{<:Index}, new_sites::AbstractVector{<:Index})
    length(old_sites) == length(new_sites) || throw(ArgumentError("Cannot reindex ITensor: site count mismatch."))
    tout = t
    for (old_s, new_s) in zip(old_sites, new_sites)
        old_s == new_s && continue
        tout = replaceind(tout, old_s, new_s)
    end
    return tout
end

# Helper function to compose TEBD gates into a single propagation ITensor
function _compose_gates_to_map(gates::AbstractVector{<:ITensor}, base_sites::AbstractVector{<:Index})
    curr_out = Dict{Index,Index}(s => prime(s) for s in base_sites)
    map_t = ITensor(1.0)
    for s in base_sites
        map_t *= delta(curr_out[s], s)
    end

    for gate in gates
        g2 = gate
        next_out = copy(curr_out)
        for s in base_sites
            hasind(g2, s) && (g2 = replaceind(g2, s, curr_out[s]))
            sp = prime(s)
            if hasind(g2, sp)
                promoted = prime(curr_out[s])
                g2 = replaceind(g2, sp, promoted)
                next_out[s] = promoted
            end
        end
        map_t = g2 * map_t
        curr_out = next_out
    end

    final_out = Index[curr_out[s] for s in base_sites]
    return map_t, final_out
end

# Helper function to create the vectorized identity ITensor for the TraceOut instrument
function _vectorized_identity_itensor(pt_sites::AbstractVector{<:Index})
    vecI = ITensor(1.0)
    for liouville_site in pt_sites
        d2 = dim(liouville_site)
        d = isqrt(d2)
        d * d == d2 || throw(ArgumentError("TraceOut requires Liouville-site dimensions d^2; got dim=$d2."))
        # Create two indices in the Hilbert space for this site
        s = Index(d, "site")
        sprime = prime(s)
        # Identity delta in Hilbert space
        deltaId = delta(s, sprime)
        # Combine (s, sprime) -> liouville_site
        cmb = combiner(s, sprime)
        # Contract, then replace with the correct Liouville site index
        Ivec = deltaId * cmb
        Ivec = replaceind(Ivec, combinedind(cmb), liouville_site)
        vecI *= Ivec
    end
    return vecI
end

# Helper functions to universally arrive at the Liouville space state for the instruments use
_coerce_liouville_state(rho0::AbstractMPS{Liouville}, sites::AbstractVector{<:Index}) =
    rho0
_coerce_liouville_state(rho0::AbstractMPO{Hilbert}, sites::AbstractVector{<:Index}) =
    to_liouville(rho0; sites=sites)
_coerce_liouville_state(rho0::AbstractMPS{Hilbert}, sites::AbstractVector{<:Index}) =
    to_liouville(to_dm(rho0); sites=sites)

function instrument_itensor(
    instr::StatePreparation,
    input_pt_sites::AbstractVector{<:Index},
    k::Int;
    kwargs...,
)
    sites = isempty(instr.pt_sites) ? Index[input_pt_sites...] : instr.pt_sites
    _validate_single_leg_sites("StatePreparation", sites, _INPUT_PLEV)
    all(s -> _tstep_from_site(s) in (nothing, k), sites) || throw(
        ArgumentError("StatePreparation: all input_pt_sites must have tstep=$k when tagged."),
    )
    stateL = _coerce_liouville_state(instr.state, sites)
    state_t = _mps_to_itensor(stateL)
    return _reindex_itensor(state_t, siteinds(stateL), sites)
end

function instrument_itensor(
    instr::ObservableMeasurement,
    output_pt_sites::AbstractVector{<:Index},
    k::Int;
    kwargs...,
)
    sites = isempty(instr.pt_sites) ? Index[output_pt_sites...] : instr.pt_sites
    _validate_single_leg_sites("ObservableMeasurement", sites, _OUTPUT_PLEV)
    all(s -> _tstep_from_site(s) in (nothing, k), sites) || throw(
        ArgumentError("ObservableMeasurement: all output_pt_sites must have tstep=$k when tagged."),
    )
    phys_sites = Index[_phys_site_from_liouv(s) for s in sites]
    obs_h = MPO(instr.op, phys_sites) # build the observable in the Hilbert space
    obs_l = to_liouville(obs_h; sites=sites) # convert the observable to the Liouville space
    return _reindex_itensor(_mps_to_itensor(obs_l), siteinds(obs_l), sites)
end

function instrument_itensor(
    instr::TraceOut,
    output_pt_sites::AbstractVector{<:Index},
    k::Int;
    kwargs...,
)
    sites = isempty(instr.pt_sites) ? Index[output_pt_sites...] : instr.pt_sites
    _validate_single_leg_sites("TraceOut", sites, _OUTPUT_PLEV)
    all(s -> _tstep_from_site(s) in (nothing, k), sites) || throw(
        ArgumentError("TraceOut: all output_pt_sites must have tstep=$k when tagged."),
    )
    return _vectorized_identity_itensor(sites)
end

function instrument_itensor(
    instr::IdentityOperation,
    input_pt_sites::AbstractVector{<:Index},
    output_pt_sites::AbstractVector{<:Index},
    k::Int;
    kwargs...,
)
    in_sites = isempty(instr.input_pt_sites) ? Index[input_pt_sites...] : instr.input_pt_sites
    out_sites = isempty(instr.output_pt_sites) ? Index[output_pt_sites...] : instr.output_pt_sites
    _validate_two_leg_map("IdentityOperation", in_sites, out_sites)
    all(s -> _tstep_from_site(s) in (nothing, k), in_sites) || throw(
        ArgumentError("IdentityOperation: all input_pt_sites must have tstep=$k when tagged."),
    )
    all(s -> _tstep_from_site(s) in (nothing, k - 1), out_sites) || throw(
        ArgumentError("IdentityOperation: all output_pt_sites must have tstep=$(k - 1) when tagged."),
    )
    map_t = ITensor(1.0)
    for (sin, sout) in zip(in_sites, out_sites)
        map_t *= delta(sin, sout)
    end
    return map_t
end

function instrument_itensor(
    instr::SystemPropagation,
    input_pt_sites::AbstractVector{<:Index},
    output_pt_sites::AbstractVector{<:Index},
    k::Int;
    dt::Real,
    order::Int=2,
    kwargs...,
)
    in_sites = isempty(instr.input_pt_sites) ? Index[input_pt_sites...] : instr.input_pt_sites
    out_sites = isempty(instr.output_pt_sites) ? Index[output_pt_sites...] : instr.output_pt_sites
    _validate_two_leg_map("SystemPropagation", in_sites, out_sites)
    all(s -> _tstep_from_site(s) in (nothing, k), in_sites) || throw(
        ArgumentError("SystemPropagation: all input_pt_sites must have tstep=$k when tagged."),
    )
    all(s -> _tstep_from_site(s) in (nothing, k - 1), out_sites) || throw(
        ArgumentError("SystemPropagation: all output_pt_sites must have tstep=$(k - 1) when tagged."),
    )
    liouv_os = OpSum_Liouville(instr.system.H, instr.system.jump_ops)
    if isempty(ITensors.terms(liouv_os))
        id_map = ITensor(1.0)
        for (sin, sout) in zip(in_sites, out_sites)
            id_map *= delta(sin, sout)
        end
        return id_map
    end
    gates = _build_trotter_gates(liouv_os, out_sites, dt; order=order)
    U_t, final_out = _compose_gates_to_map(gates, out_sites)
    return _reindex_itensor(U_t, final_out, in_sites)
end

function instrument_itensor(
    instr::AbstractInstrument,
    args...;
    kwargs...,
)
    println("No instrument_itensor constructor is defined for $(typeof(instr)). Future dev will allow user-defined abstract instrument types.")
end

end # module Instruments
