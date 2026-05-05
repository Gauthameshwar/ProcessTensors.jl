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
       IdentityOperation, SystemPropagation, resolve_instrument,
       InstrumentSeq, add!, instrument_itensor, instrument_leg_maps

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
# InstrumentSeq — unified schedule (default + per-tstep entries + bounds)
# =========================================================================

mutable struct InstrumentSeq
    default::AbstractInstrument
    entries::Dict{Int,AbstractInstrument}
    nsteps::Int # upper bound for validation; 0 = unchecked until bound to a ProcessTensor
end

function InstrumentSeq(
    default::AbstractInstrument,
    nsteps::Int=0;
    init::Union{Nothing,StatePreparation}=nothing,
    overrides::AbstractDict{Int,<:AbstractInstrument}=Dict{Int,AbstractInstrument}(),
    entries::Union{Nothing,AbstractDict{Int,<:AbstractInstrument}}=nothing,
)
    d = if entries === nothing
        Dict{Int,AbstractInstrument}(pairs(overrides))
    else
        Dict{Int,AbstractInstrument}(pairs(entries))
    end
    if init !== nothing
        d[0] = init
    end
    return InstrumentSeq(default, d, nsteps)
end

"""
    InstrumentSeq(; default, nsteps=0, entries...)

Empty schedule with a fallback `default` instrument and optional `entries` dictionary.
"""
function InstrumentSeq(; default::AbstractInstrument, nsteps::Int=0, entries=Dict{Int,AbstractInstrument}())
    return InstrumentSeq(default, nsteps; entries=entries)
end

"""
    resolve_instrument(seq::InstrumentSeq, k::Int) -> Union{AbstractInstrument,Nothing}

- `k == 0`: `entries[0]` if set (typically `StatePreparation`), else `nothing`.
- `k ≥ 1`: `entries[k]` if set, else `seq.default`.
"""
function resolve_instrument(seq::InstrumentSeq, k::Int)
    k == 0 && return get(seq.entries, 0, nothing)
    k >= 1 || throw(ArgumentError("resolve_instrument: expected k ≥ 0; got $k."))
    return get(seq.entries, k, seq.default)
end

"""
    resolve_instrument(seq::InstrumentSeq, k::Int, fallback::AbstractInstrument)

Same as `resolve_instrument(seq, k)` for `k == 0`, but for `k ≥ 1` uses `fallback`
when `entries[k]` is absent (e.g. evolve-time default override).
"""
function resolve_instrument(seq::InstrumentSeq, k::Int, fallback::AbstractInstrument)
    k == 0 && return get(seq.entries, 0, nothing)
    k >= 1 || throw(ArgumentError("resolve_instrument: expected k ≥ 0; got $k."))
    return get(seq.entries, k, fallback)
end

"""
    add!(seq::InstrumentSeq, instr::AbstractInstrument, tstep::Int) -> seq

Replace or insert the instrument at logical timestep `tstep`.
`nsteps` upper bound is checked when `seq.nsteps > 0`.

Constraints:
- `tstep ≥ 0`
- `tstep == 0` only allows [`StatePreparation`](@ref).
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
    if seq.nsteps > 0 && tstep > seq.nsteps
        throw(ArgumentError("add!: tstep=$tstep exceeds seq.nsteps=$(seq.nsteps)."))
    end
    seq.entries[tstep] = instr
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
    ks = sort!(collect(keys(seq.entries)))
    print(io, "InstrumentSeq(default=$(typeof(seq.default)), nsteps=$(seq.nsteps), $(length(ks)) explicit entries)")
    for k in ks
        print(io, "\n  tstep=$k => ", typeof(seq.entries[k]))
    end
end

"""
    instrument_leg_maps(seq::InstrumentSeq, nsteps::Int) -> (in_map, out_map, missing_in, missing_out)

PT leg convention matches [`coupling_times`](@ref) evolve slots `step ∈ 1:nsteps`:
primed input at `tstep = step`, unprimed output at `tstep = step-1`, except the
terminal primed leg `tstep = nsteps` and terminal unprimed leg `tstep = nsteps-1`
are not required in the maps.

`missing_out` only lists `tstep = 0 … nsteps-2`; the final open output leg
` tstep = nsteps-1 ` may be absent.
"""
function instrument_leg_maps(seq::InstrumentSeq, nsteps::Int)
    nsteps >= 1 || throw(ArgumentError("instrument_leg_maps: nsteps must be >= 1"))

    in_map = Dict{Int,AbstractInstrument}()
    out_map = Dict{Int,AbstractInstrument}()

    for step in 1:nsteps
        instr = resolve_instrument(seq, step)
        if instr isa TwoLegInstrument
            if step <= nsteps - 1
                in_map[step] = instr
            end
            tout = step - 1
            if tout <= nsteps - 2
                out_map[tout] = instr
            end
        elseif instr isa ObservableMeasurement || instr isa TraceOut
            if instr.leg_plev == _OUTPUT_PLEV
                tout = step - 1
                if tout <= nsteps - 2
                    out_map[tout] = instr
                end
            else
                if step <= nsteps - 1
                    in_map[step] = instr
                end
            end
        elseif instr isa StatePreparation
            throw(
                ArgumentError(
                    "instrument_leg_maps: StatePreparation is only valid at tstep=0, not at evolve slot step=$step.",
                ),
            )
        end
    end

    prep = resolve_instrument(seq, 0)
    if prep !== nothing
        prep isa StatePreparation || throw(ArgumentError("instrument_leg_maps: tstep=0 must be StatePreparation"))
        in_map[0] = prep
    end

    expected_in = collect(0:nsteps-1)
    expected_out = nsteps == 1 ? Int[] : collect(0:nsteps-2)
    missing_in = [k for k in expected_in if !haskey(in_map, k)]
    missing_out = [k for k in expected_out if !haskey(out_map, k)]

    return in_map, out_map, missing_in, missing_out
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
    pt_sites_arg::AbstractVector{<:Index},
    k::Int;
    kwargs...,
)
    sites = isempty(instr.pt_sites) ? Index[pt_sites_arg...] : instr.pt_sites
    _validate_single_leg_sites("StatePreparation", sites, instr.leg_plev)
    all(s -> _tstep_from_site(s) in (nothing, k), sites) || throw(
        ArgumentError("StatePreparation: all pt_sites must have tstep=$k when tagged."),
    )
    stateL = _coerce_liouville_state(instr.state, sites)
    state_t = _mps_to_itensor(stateL)
    return _reindex_itensor(state_t, siteinds(stateL), sites)
end

function instrument_itensor(
    instr::ObservableMeasurement,
    pt_sites_arg::AbstractVector{<:Index},
    k::Int;
    kwargs...,
)
    sites = isempty(instr.pt_sites) ? Index[pt_sites_arg...] : instr.pt_sites
    _validate_single_leg_sites("ObservableMeasurement", sites, instr.leg_plev)
    all(s -> _tstep_from_site(s) in (nothing, k), sites) || throw(
        ArgumentError("ObservableMeasurement: all pt_sites must have tstep=$k when tagged."),
    )
    phys_sites = Index[_phys_site_from_liouv(s) for s in sites]
    obs_h = MPO(instr.op, phys_sites) # build the observable in the Hilbert space
    obs_l = to_liouville(obs_h; sites=sites) # convert the observable to the Liouville space
    return _reindex_itensor(_mps_to_itensor(obs_l), siteinds(obs_l), sites)
end

function instrument_itensor(
    instr::TraceOut,
    pt_sites_arg::AbstractVector{<:Index},
    k::Int;
    kwargs...,
)
    sites = isempty(instr.pt_sites) ? Index[pt_sites_arg...] : instr.pt_sites
    _validate_single_leg_sites("TraceOut", sites, instr.leg_plev)
    all(s -> _tstep_from_site(s) in (nothing, k), sites) || throw(
        ArgumentError("TraceOut: all pt_sites must have tstep=$k when tagged."),
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
    throw(MethodError(instrument_itensor, (instr, args...)))
end

end # module Instruments
