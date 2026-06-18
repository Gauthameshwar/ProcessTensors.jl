module Instruments

import ..ProcessTensors
import ..ProcessTensors: add!
using ITensors
using LinearAlgebra
using ..ProcessTensors: AbstractMPO, AbstractMPS, AbstractSystem, Hilbert, Liouville, MPO,
                        OpSum, OpSum_Liouville, Index, ITensor, apply, dim, plev, prime,
                        replaceind, siteinds, tag_value, to_dm, to_liouville,
                        _phys_site_from_liouv, _superop_matrix, _LiouvLeft, _LiouvRight,
                        liouvillian_propagator_itensor, Exact, Trotter

export AbstractInstrument, SingleLegInstrument, TwoLegInstrument,
       StatePreparation, ObservableMeasurement, TraceOut,
       IdentityOperation, SystemPropagation, OpenOutput, ProductInstrument, CustomTwoLegInstrument,
       LeftRightOperator, left_action, right_action, 
       resolve_instrument, InstrumentSeq, add!, instrument_itensor, instrument_leg_maps

abstract type AbstractInstrument end
abstract type SingleLegInstrument <: AbstractInstrument end
abstract type TwoLegInstrument <: AbstractInstrument end

# Prime level convention: input legs are primed (plev=1), output legs are unprimed (plev=0)
const _INPUT_PLEV = 1
const _OUTPUT_PLEV = 0

# Validate the prime level of a leg. It should be either 0 or 1
_assert_valid_leg_plev(leg_plev::Int) =
    leg_plev in (_INPUT_PLEV, _OUTPUT_PLEV) || throw(
        ArgumentError("Instrument leg prime level must be 0 (output) or 1 (input); got $leg_plev."),
    )

# Validate the single-leg instrument has exactly one site and that it has the correct prime level
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

# Get the tstep from a site.
_tstep_from_site(s::Index) = begin
    tstep_str = tag_value(s, "tstep=")
    return tstep_str === nothing ? nothing : parse(Int, tstep_str)
end

# Validate the two-leg instrument has exactly one input and one output site and that they have the correct prime levels
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

struct StatePreparation{M<:Union{AbstractMPS,AbstractMPO{Hilbert}}} <: SingleLegInstrument
    state::M
    pt_sites::Vector{Index}
    leg_plev::Int
end
function StatePreparation(
    state::Union{AbstractMPS,AbstractMPO{Hilbert}},
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

struct _ComposedSingleLegInstrument{F<:Tuple} <: SingleLegInstrument
    factors::F
    pt_sites::Vector{Index}
    leg_plev::Int
end

"""
    LeftRightOperator(A, B)

Two-leg instrument implementing ``\\rho \\mapsto A\\,\\rho\\,B`` on the system Hilbert factor,
equivalently ``\\mathrm{vec}(\\rho) \\mapsto (B^{\\mathsf T} \\otimes A)\\,\\mathrm{vec}(\\rho)`` on
the Liouville leg. Both arguments are `MPO{Hilbert}` on the same physical sites.

Use [`left_action`](@ref) / [`right_action`](@ref) for the common cases ``O\\rho`` and ``\\rho O``.
"""
struct LeftRightOperator{A<:AbstractMPO{Hilbert},B<:AbstractMPO{Hilbert}} <: TwoLegInstrument
    left::A
    right::B
    input_pt_sites::Vector{Index}
    output_pt_sites::Vector{Index}
end

function LeftRightOperator(
    left::AbstractMPO{Hilbert},
    right::AbstractMPO{Hilbert},
    input_pt_sites::AbstractVector{<:Index}=Index[],
    output_pt_sites::AbstractVector{<:Index}=Index[],
)
    left_sites = _phys_sites_from_hilbert_mpo(left)
    right_sites = _phys_sites_from_hilbert_mpo(right)
    left_sites == right_sites || throw(
        ArgumentError("LeftRightOperator: left and right MPOs must share the same siteinds."),
    )
    input_vec = Index[input_pt_sites...]
    output_vec = Index[output_pt_sites...]
    if !isempty(input_vec) || !isempty(output_vec)
        _validate_two_leg_map("LeftRightOperator", input_vec, output_vec)
    end
    return LeftRightOperator(left, right, input_vec, output_vec)
end

function _identity_hilbert_mpo(phys_sites::AbstractVector{<:Index})
    os = OpSum()
    for j in eachindex(phys_sites)
        os += 1.0, "Id", j
    end
    return MPO(os, phys_sites)
end

function _fold_observable_factors(factors, phys_sites::AbstractVector{<:Index})
    op_acc = nothing
    for f in factors
        f isa ObservableMeasurement || continue
        O_mpo = MPO(f.op, phys_sites)
        op_acc = op_acc === nothing ? O_mpo : apply(O_mpo, op_acc)
    end
    op_acc === nothing && throw(ArgumentError("Composed instrument has no ObservableMeasurement factors."))
    return op_acc
end

function _composed_observable_mpo(composed::_ComposedSingleLegInstrument, phys_sites::AbstractVector{<:Index})
    return _fold_observable_factors(composed.factors, phys_sites)
end

function _phys_sites_from_hilbert_mpo(mpo::AbstractMPO{Hilbert})
    return Index[
        only(filter(i -> plev(i) == _OUTPUT_PLEV, inds(mpo.core[j])))
        for j in eachindex(mpo.core)
    ]
end

"""``\\rho \\mapsto A\\rho`` (identity on the right)."""
function left_action(A::AbstractMPO{Hilbert})
    return LeftRightOperator(A, _identity_hilbert_mpo(_phys_sites_from_hilbert_mpo(A)))
end

function left_action(O::OpSum, phys_sites::AbstractVector{<:Index})
    return left_action(MPO(O, phys_sites))
end

function left_action(composed::_ComposedSingleLegInstrument, phys_sites::AbstractVector{<:Index})
    return left_action(_composed_observable_mpo(composed, phys_sites))
end

"""``\\rho \\mapsto \\rho B`` (identity on the left)."""
function right_action(B::AbstractMPO{Hilbert})
    return LeftRightOperator(_identity_hilbert_mpo(_phys_sites_from_hilbert_mpo(B)), B)
end

function right_action(O::OpSum, phys_sites::AbstractVector{<:Index})
    return right_action(MPO(O, phys_sites))
end

function right_action(composed::_ComposedSingleLegInstrument, phys_sites::AbstractVector{<:Index})
    return right_action(_composed_observable_mpo(composed, phys_sites))
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

"""
    OpenOutput

Two-leg instrument for a causality cut at evolve slot `s`: apply `vec(I)` on the
primed input `in_s` and leave the previous unprimed output `out_{s-1}` open (the
returned ITensor carries only `in_s`, not `out_{s-1}`).
"""
struct OpenOutput <: TwoLegInstrument
    input_pt_sites::Vector{Index}
    output_pt_sites::Vector{Index}
end
function OpenOutput(
    input_pt_sites::AbstractVector{<:Index},
    output_pt_sites::AbstractVector{<:Index},
)
    input_vec = Index[input_pt_sites...]
    output_vec = Index[output_pt_sites...]
    _validate_two_leg_map("OpenOutput", input_vec, output_vec)
    return OpenOutput(input_vec, output_vec)
end
OpenOutput() = OpenOutput(Index[], Index[])

"""
    ProductInstrument

Two-leg instrument at one evolve slot: an output-leg factor (`plev = 0`, time `step - 1`)
and an input-leg factor (`plev = 1`, time `step`). Construct via `output_factor * input_factor`
(order-independent for single-leg factors).
"""
struct ProductInstrument{I<:SingleLegInstrument,O<:SingleLegInstrument} <: TwoLegInstrument
    input_instr::I
    output_instr::O
end

function Base.show(io::IO, instr::ProductInstrument)
    print(io, instr.output_instr, " * ", instr.input_instr)
end

"""
    CustomTwoLegInstrument

Two-leg instrument backed by a dense `ITensor` on Liouville process-tensor legs.

Construct in either of two ways:

- **Ready tensor:** `CustomTwoLegInstrument(data, input_pt_sites, output_pt_sites)` when
  `data` is already an `instrument_itensor` (or equivalent) with indices matching the
  supplied PT legs.

- **Reindexing tensor:** `CustomTwoLegInstrument(data; source_input=..., source_output=...,
  input_pt_sites=..., output_pt_sites=...)` when `data` carries source indices that are
  replaced by the target PT legs at contraction time. Leave `input_pt_sites` /
  `output_pt_sites` empty for lazy PT-leg binding (same convention as
  [`IdentityOperation`](@ref)).
"""
struct CustomTwoLegInstrument <: TwoLegInstrument
    data::ITensor
    input_pt_sites::Vector{Index}
    output_pt_sites::Vector{Index}
    source_input::Vector{Index}
    source_output::Vector{Index}
end

function _validate_custom_data_indices(data::ITensor, input_sites::AbstractVector{<:Index}, output_sites::AbstractVector{<:Index})
    for s in input_sites
        hasind(data, s) || throw(
            ArgumentError("CustomTwoLegInstrument: data is missing input index $s."),
        )
    end
    for s in output_sites
        hasind(data, s) || throw(
            ArgumentError("CustomTwoLegInstrument: data is missing output index $s."),
        )
    end
    expected = length(input_sites) + length(output_sites)
    length(inds(data)) == expected || throw(
        ArgumentError(
            "CustomTwoLegInstrument: data must have exactly $(expected) indices; got $(length(inds(data))).",
        ),
    )
    return nothing
end

function CustomTwoLegInstrument(
    data::ITensor,
    input_pt_sites::AbstractVector{<:Index},
    output_pt_sites::AbstractVector{<:Index},
)
    in_vec = Index[input_pt_sites...]
    out_vec = Index[output_pt_sites...]
    _validate_two_leg_map("CustomTwoLegInstrument", in_vec, out_vec)
    _validate_custom_data_indices(data, in_vec, out_vec)
    return CustomTwoLegInstrument(data, in_vec, out_vec, Index[], Index[])
end

function CustomTwoLegInstrument(
    data::ITensor;
    source_input::AbstractVector{<:Index},
    source_output::AbstractVector{<:Index},
    input_pt_sites::AbstractVector{<:Index}=Index[],
    output_pt_sites::AbstractVector{<:Index}=Index[],
)
    src_in = Index[source_input...]
    src_out = Index[source_output...]
    in_vec = Index[input_pt_sites...]
    out_vec = Index[output_pt_sites...]
    if !isempty(in_vec) && length(src_in) != length(in_vec)
        throw(
            ArgumentError(
                "CustomTwoLegInstrument: source_input and input_pt_sites must have equal length; " *
                "got $(length(src_in)) and $(length(in_vec)).",
            ),
        )
    end
    if !isempty(out_vec) && length(src_out) != length(out_vec)
        throw(
            ArgumentError(
                "CustomTwoLegInstrument: source_output and output_pt_sites must have equal length; " *
                "got $(length(src_out)) and $(length(out_vec)).",
            ),
        )
    end
    if !isempty(in_vec) || !isempty(out_vec)
        _validate_two_leg_map("CustomTwoLegInstrument", in_vec, out_vec)
    end
    _validate_custom_data_indices(data, src_in, src_out)
    return CustomTwoLegInstrument(data, in_vec, out_vec, src_in, src_out)
end

function Base.show(io::IO, instr::CustomTwoLegInstrument)
    print(io, "CustomTwoLegInstrument(")
    if isempty(instr.source_input)
        print(io, "ready, nind=", length(inds(instr.data)), ")")
    else
        print(io, "reindex, nind=", length(inds(instr.data)), ")")
    end
end
function Base.:(*)(a::SingleLegInstrument, b::SingleLegInstrument)
    if a.leg_plev == _OUTPUT_PLEV && b.leg_plev == _INPUT_PLEV
        (a isa StatePreparation || b isa StatePreparation) && throw(
            ArgumentError("Cannot multiply StatePreparation into a ProductInstrument."),
        )
        return ProductInstrument(b, a)
    elseif a.leg_plev == _INPUT_PLEV && b.leg_plev == _OUTPUT_PLEV
        (a isa StatePreparation || b isa StatePreparation) && throw(
            ArgumentError("Cannot multiply StatePreparation into a ProductInstrument."),
        )
        return ProductInstrument(a, b)
    elseif a.leg_plev == b.leg_plev
        _is_a_valid_product_instrument(a::SingleLegInstrument) = a isa ObservableMeasurement || a isa StatePreparation || a isa _ComposedSingleLegInstrument

        # Proceed only if both factors are valid product instruments
        (_is_a_valid_product_instrument(a) && _is_a_valid_product_instrument(b)) || throw(
            ArgumentError("Same-leg instrument products only support ObservableMeasurement and StatePreparation factors."),
        )

        factors = (
            (a isa _ComposedSingleLegInstrument ? a.factors : (a,))...,
            (b isa _ComposedSingleLegInstrument ? b.factors : (b,))...,
        )
        # Check to ensure we don't have multiple state preparations
        nprep = count(f -> f isa StatePreparation, factors)
        nprep <= 1 || throw(ArgumentError("Same-leg instrument products support at most one StatePreparation."))
        if nprep == 1 && !(first(factors) isa StatePreparation || last(factors) isa StatePreparation)
            throw(ArgumentError("StatePreparation must be the first or final factor in a same-leg instrument product."))
        end

        a_sites = a.pt_sites
        b_sites = b.pt_sites
        pt_sites = if isempty(a_sites)
            Index[b_sites...]
        elseif isempty(b_sites)
            Index[a_sites...]
        elseif a_sites == b_sites
            Index[a_sites...]
        else
            throw(ArgumentError("Same-leg instrument products require matching pt_sites when both factors are bound."))
        end
        return _ComposedSingleLegInstrument(factors, pt_sites, a.leg_plev)
    end
    throw(
        ArgumentError(
            "Product of single-leg instruments requires one output (plev=0) and one input " *
            "(plev=1); got plev=($(a.leg_plev), $(b.leg_plev)).",
        ),
    )
end

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
    valid_init = instr isa StatePreparation ||
                 (instr isa _ComposedSingleLegInstrument && any(f -> f isa StatePreparation, instr.factors))
    if tstep == 0 && !valid_init
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
        elseif instr isa ObservableMeasurement ||
               instr isa _ComposedSingleLegInstrument ||
               instr isa TraceOut
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
        prep isa StatePreparation ||
            (prep isa _ComposedSingleLegInstrument && any(f -> f isa StatePreparation, prep.factors)) ||
            throw(ArgumentError("instrument_leg_maps: tstep=0 must be StatePreparation"))
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
    instr::_ComposedSingleLegInstrument,
    pt_sites_arg::AbstractVector{<:Index},
    k::Int;
    kwargs...,
)
    sites = isempty(instr.pt_sites) ? Index[pt_sites_arg...] : instr.pt_sites
    _validate_single_leg_sites("_ComposedSingleLegInstrument", sites, instr.leg_plev)
    all(s -> _tstep_from_site(s) in (nothing, k), sites) || throw(
        ArgumentError("_ComposedSingleLegInstrument: all pt_sites must have tstep=$k when tagged."),
    )
    ρ = nothing
    for f in instr.factors
        if f isa StatePreparation
            ρ = if f.state isa AbstractMPO{Hilbert}
                f.state
            elseif f.state isa AbstractMPS{Hilbert}
                to_dm(f.state)
            else
                throw(ArgumentError("_ComposedSingleLegInstrument: StatePreparation state must be Hilbert MPS or MPO."))
            end
        end
    end
    phys_sites = ρ === nothing ?
                 Index[_phys_site_from_liouv(s) for s in sites] :
                 _phys_sites_from_hilbert_mpo(ρ)
    has_obs = any(f -> f isa ObservableMeasurement, instr.factors)
    op_acc = has_obs ? _fold_observable_factors(instr.factors, phys_sites) : nothing
    prep_first = any(f -> f isa StatePreparation, instr.factors) && first(instr.factors) isa StatePreparation
    hilbert_mpo = if ρ === nothing
        op_acc
    elseif op_acc === nothing
        ρ
    elseif prep_first
        apply(ρ, op_acc)
    else
        apply(op_acc, ρ)
    end
    hilbert_mpo === nothing && throw(ArgumentError("_ComposedSingleLegInstrument: no factors to build."))
    state_l = to_liouville(hilbert_mpo; sites=sites)
    return _reindex_itensor(_mps_to_itensor(state_l), siteinds(state_l), sites)
end

function instrument_itensor(
    instr::LeftRightOperator,
    input_pt_sites::AbstractVector{<:Index},
    output_pt_sites::AbstractVector{<:Index},
    k::Int;
    kwargs...,
)
    in_sites = isempty(instr.input_pt_sites) ? Index[input_pt_sites...] :
instr.input_pt_sites
    out_sites = isempty(instr.output_pt_sites) ? Index[output_pt_sites...] :
instr.output_pt_sites
    _validate_two_leg_map("LeftRightOperator", in_sites, out_sites)

    inp = only(in_sites)
    out = only(out_sites)

    # Extract physical sites and convert MPOs to matrices
    phys_sites = _phys_sites_from_hilbert_mpo(instr.left)
    _phys_sites_from_hilbert_mpo(instr.right) == phys_sites || throw(
        ArgumentError("LeftRightOperator: left and right MPOs must share physical sites."),
    )

    T_A = _mpo_to_itensor(instr.left)
    T_B = _mpo_to_itensor(instr.right)
    d = prod(dim.(phys_sites))

    # Extract as matrices
    A_mat = reshape(ComplexF64.(Array(T_A, prime.(phys_sites)..., phys_sites...)), d, d)
    B_mat = reshape(ComplexF64.(Array(T_B, prime.(phys_sites)..., phys_sites...)), d, d)

    # Use existing _superop_matrix helpers to build Liouville embeddings
    Id = Matrix{ComplexF64}(I, d, d)
    left_superop = _superop_matrix(_LiouvLeft(), A_mat, Id)    # I ⊗ A
    right_superop = _superop_matrix(_LiouvRight(), B_mat, Id)  # B^T ⊗ I

    # Compose superoperators: (B^T ⊗ I) * (I ⊗ A) = B^T ⊗ A
    W = right_superop * left_superop

    # Return ITensor with BOTH indices properly bound
    return ITensor(reshape(W, d^2, d^2), inp, out)
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
    instr::ProductInstrument,
    input_pt_sites::AbstractVector{<:Index},
    output_pt_sites::AbstractVector{<:Index},
    k::Int;
    kwargs...,
)
    T_out = instrument_itensor(instr.output_instr, output_pt_sites, k - 1; kwargs...)
    T_in = instrument_itensor(instr.input_instr, input_pt_sites, k; kwargs...)
    return T_in * T_out
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
    instr::OpenOutput,
    input_pt_sites::AbstractVector{<:Index},
    output_pt_sites::AbstractVector{<:Index},
    k::Int;
    kwargs...,
)
    in_sites = isempty(instr.input_pt_sites) ? Index[input_pt_sites...] : instr.input_pt_sites
    out_sites = isempty(instr.output_pt_sites) ? Index[output_pt_sites...] : instr.output_pt_sites
    _validate_two_leg_map("OpenOutput", in_sites, out_sites)
    all(s -> _tstep_from_site(s) in (nothing, k), in_sites) || throw(
        ArgumentError("OpenOutput: all input_pt_sites must have tstep=$k when tagged."),
    )
    all(s -> _tstep_from_site(s) in (nothing, k - 1), out_sites) || throw(
        ArgumentError("OpenOutput: all output_pt_sites must have tstep=$(k - 1) when tagged."),
    )
    return _vectorized_identity_itensor(in_sites)
end

function instrument_itensor(
    instr::SystemPropagation,
    input_pt_sites::AbstractVector{<:Index},
    output_pt_sites::AbstractVector{<:Index},
    k::Int;
    dt::Real,
    alg=Trotter{2}(),
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
    # liouvillian_propagator_itensor builds on canonical system Liouville sites (with `Site`
    # tag). PT legs drop `Site` to stay within ITensors' four-tag limit once `tstep=` is added.
    gate_sites = Index[instr.system.sites...]
    length(gate_sites) == length(out_sites) || throw(
        ArgumentError(
            "SystemPropagation: expected $(length(out_sites)) system sites, got $(length(gate_sites)).",
        ),
    )
    U_t = liouvillian_propagator_itensor(
        instr.system.H,
        gate_sites,
        dt;
        alg=alg,
        jump_ops=instr.system.jump_ops,
    )
    # U_t has unprimed gate_sites (output) and prime.(gate_sites) (input).
    # Relabel to PT leg convention: output → out_sites, input → in_sites.
    U_pt = U_t
    for (g, pt_out) in zip(gate_sites, out_sites)
        U_pt = replaceind(U_pt, g, pt_out)
    end
    for (g, pt_in) in zip(gate_sites, in_sites)
        U_pt = replaceind(U_pt, prime(g), pt_in)
    end
    return U_pt
end

function instrument_itensor(
    instr::CustomTwoLegInstrument,
    input_pt_sites::AbstractVector{<:Index},
    output_pt_sites::AbstractVector{<:Index},
    k::Int;
    kwargs...,
)
    runtime_in = Index[input_pt_sites...]
    runtime_out = Index[output_pt_sites...]
    _validate_two_leg_map("CustomTwoLegInstrument", runtime_in, runtime_out)
    all(s -> _tstep_from_site(s) in (nothing, k), runtime_in) || throw(
        ArgumentError("CustomTwoLegInstrument: all input_pt_sites must have tstep=$k when tagged."),
    )
    all(s -> _tstep_from_site(s) in (nothing, k - 1), runtime_out) || throw(
        ArgumentError("CustomTwoLegInstrument: all output_pt_sites must have tstep=$(k - 1) when tagged."),
    )

    if isempty(instr.source_input) && isempty(instr.source_output)
        return _reindex_itensor(
            _reindex_itensor(copy(instr.data), instr.input_pt_sites, runtime_in),
            instr.output_pt_sites,
            runtime_out,
        )
    end

    target_in = isempty(instr.input_pt_sites) ? runtime_in : instr.input_pt_sites
    target_out = isempty(instr.output_pt_sites) ? runtime_out : instr.output_pt_sites
    _validate_two_leg_map("CustomTwoLegInstrument", target_in, target_out)

    t = _reindex_itensor(
        _reindex_itensor(copy(instr.data), instr.source_input, target_in),
        instr.source_output,
        target_out,
    )
    return _reindex_itensor(_reindex_itensor(t, target_in, runtime_in), target_out, runtime_out)
end
function instrument_itensor(
    instr::AbstractInstrument,
    args...;
    kwargs...,
)
    throw(MethodError(instrument_itensor, (instr, args...)))
end

end # module Instruments
