# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: src/instruments/lazy_instruments.jl
# Contributor: Gauthameshwar S.
#
# Defines lazy instrument types, constructors, composition, and schedules for
# process-tensor contractions.

"""
    AbstractInstrument

Abstract interface for operations inserted on process-tensor legs.

Instruments represent state preparations, measurements, traces, identity
connectors, explicit unitary control maps, and custom Liouville maps used by
[`evaluate_process`](@ref ProcessTensors.evaluate_process) and
[`evolve`](@ref ProcessTensors.evolve).
"""
abstract type AbstractInstrument end

"""
    SingleLegInstrument

Instrument acting on one process-tensor leg.

Single-leg instruments are bound either to a primed input leg (`plev = 1`) or an
unprimed output leg (`plev = 0`). Examples include [`StatePreparation`](@ref),
[`ObservableMeasurement`](@ref), and [`TraceOut`](@ref).
"""
abstract type SingleLegInstrument <: AbstractInstrument end

"""
    TwoLegInstrument

Instrument connecting one primed input process-tensor leg to one unprimed output leg.

Two-leg instruments represent maps between adjacent evolve slots, such as
[`IdentityOperation`](@ref), [`UnitaryPropagation`](@ref), [`LeftRightOperator`](@ref),
and custom Liouville superoperators.
"""
abstract type TwoLegInstrument <: AbstractInstrument end

# Prime level convention: input legs are primed (plev=1), output legs are unprimed (plev=0)
const _INPUT_PLEV = 1
const _OUTPUT_PLEV = 0

# Instrument legs are either unprimed outputs (`plev=0`) or primed inputs (`plev=1`).
_assert_valid_leg_plev(leg_plev::Int) =
    leg_plev in (_INPUT_PLEV, _OUTPUT_PLEV) || throw(
        ArgumentError("Instrument leg prime level must be 0 (output) or 1 (input); got $leg_plev."),
    )

# Single-leg instruments bind to exactly one PT site with the requested prime level.
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

# Two-leg maps connect input `tstep=k` to output `tstep=k-1`.
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

# Collect single-leg PT sites into a Vector{Index}, validating non-empty bindings.
# Empty inputs stay empty for lazy PT-leg binding at contraction time.
function _bind_single_leg_sites(instr_name::AbstractString, pt_sites, leg_plev::Int)
    pt_sites_vec = Index[pt_sites...]
    isempty(pt_sites_vec) || _validate_single_leg_sites(instr_name, pt_sites_vec, leg_plev)
    return pt_sites_vec
end

# Collect two-leg PT sites into (input, output) Vector{Index} pairs, validating the leg map.
# With `lazy_ok=true`, fully-empty bindings skip validation for deferred PT-leg binding.
function _bind_two_leg_sites(instr_name::AbstractString, input_pt_sites, output_pt_sites; lazy_ok::Bool=false)
    input_vec = Index[input_pt_sites...]
    output_vec = Index[output_pt_sites...]
    if !lazy_ok || !isempty(input_vec) || !isempty(output_vec)
        _validate_two_leg_map(instr_name, input_vec, output_vec)
    end
    return input_vec, output_vec
end

"""
    StatePreparation

Single-leg instrument that prepares a system state on a process-tensor leg.

Canonical construction uses already-normalized fields:

```julia
StatePreparation(state, pt_sites::Vector{Index}, leg_plev::Int)
```

Prefer [`state_preparation`](@ref) for ordinary use.
"""
struct StatePreparation{M<:Union{AbstractMPS,AbstractMPO{Hilbert}}} <: SingleLegInstrument
    state::M
    pt_sites::Vector{Index}
    leg_plev::Int

    function StatePreparation(
        state::M,
        pt_sites::Vector{Index},
        leg_plev::Int,
    ) where {M<:Union{AbstractMPS,AbstractMPO{Hilbert}}}
        leg_plev == _INPUT_PLEV || throw(
            ArgumentError("StatePreparation is fixed to leg_plev=1 (input leg); got leg_plev=$leg_plev."),
        )
        isempty(pt_sites) || _validate_single_leg_sites("StatePreparation", pt_sites, _INPUT_PLEV)
        return new{M}(state, pt_sites, _INPUT_PLEV)
    end
end

"""
    state_preparation(state, pt_sites=Index[]; leg_plev=1)

User-facing constructor for [`StatePreparation`](@ref).

Hilbert inputs are converted to Liouville space when the instrument tensor is
materialized. Leave `pt_sites` empty for lazy binding in
[`create_instruments`](@ref ProcessTensors.Instruments.create_instruments).

# Examples
```julia
seq = InstrumentSeq(default=identity_operation(), nsteps=pt.nsteps)
add!(seq, state_preparation(ρ0), 0)
```
"""
function state_preparation(
    state::Union{AbstractMPS,AbstractMPO{Hilbert}},
    pt_sites::AbstractVector{<:Index}=Index[];
    leg_plev::Int=_INPUT_PLEV,
)
    pt_sites_vec = _bind_single_leg_sites("StatePreparation", pt_sites, _INPUT_PLEV)
    return StatePreparation(state, pt_sites_vec, leg_plev)
end

"""
    ObservableMeasurement

Single-leg instrument representing insertion of a Hilbert-space observable.

Canonical construction uses already-normalized fields:

```julia
ObservableMeasurement(op, pt_sites::Vector{Index}, leg_plev::Int)
```

Prefer [`observable_measurement`](@ref) for ordinary use.
"""
struct ObservableMeasurement{O<:OpSum} <: SingleLegInstrument
    op::O
    pt_sites::Vector{Index}
    leg_plev::Int

    function ObservableMeasurement(
        op::O,
        pt_sites::Vector{Index},
        leg_plev::Int,
    ) where {O<:OpSum}
        _assert_valid_leg_plev(leg_plev)
        isempty(pt_sites) || _validate_single_leg_sites("ObservableMeasurement", pt_sites, leg_plev)
        return new{O}(op, pt_sites, leg_plev)
    end
end

"""
    observable_measurement(op::OpSum, pt_sites=Index[]; leg_plev=0)

User-facing constructor for [`ObservableMeasurement`](@ref).

`leg_plev=0` targets an output leg; `leg_plev=1` targets an input leg for
right-action correlation schedules.

# Examples
```julia
O = OpSum()
O += 1.0, "Sz", 1
add!(seq, observable_measurement(O), 2)
```
"""
function observable_measurement(
    op::OpSum,
    pt_sites::AbstractVector{<:Index}=Index[];
    leg_plev::Int=_OUTPUT_PLEV,
)
    pt_sites_vec = _bind_single_leg_sites("ObservableMeasurement", pt_sites, leg_plev)
    return ObservableMeasurement(op, pt_sites_vec, leg_plev)
end

struct _ComposedSingleLegInstrument{F<:Tuple} <: SingleLegInstrument
    factors::F
    pt_sites::Vector{Index}
    leg_plev::Int
end

"""
    LeftRightOperator

Two-leg instrument implementing the Liouville-space map ``\\rho \\mapsto A\\rho B``.

Canonical construction uses already-normalized fields:

```julia
LeftRightOperator(left, right, input_pt_sites::Vector{Index}, output_pt_sites::Vector{Index})
```

Prefer [`left_right_operator`](@ref) for ordinary use.
"""
struct LeftRightOperator{A<:AbstractMPO{Hilbert},B<:AbstractMPO{Hilbert}} <: TwoLegInstrument
    left::A
    right::B
    input_pt_sites::Vector{Index}
    output_pt_sites::Vector{Index}

    function LeftRightOperator(
        left::A,
        right::B,
        input_pt_sites::Vector{Index},
        output_pt_sites::Vector{Index},
    ) where {A<:AbstractMPO{Hilbert},B<:AbstractMPO{Hilbert}}
        left_sites = _phys_sites_from_hilbert_mpo(left)
        right_sites = _phys_sites_from_hilbert_mpo(right)
        left_sites == right_sites || throw(
            ArgumentError("LeftRightOperator: left and right MPOs must share the same siteinds."),
        )
        if !isempty(input_pt_sites) || !isempty(output_pt_sites)
            _validate_two_leg_map("LeftRightOperator", input_pt_sites, output_pt_sites)
        end
        return new{A,B}(left, right, input_pt_sites, output_pt_sites)
    end
end

"""
    left_right_operator(left, right, input_pt_sites=Index[], output_pt_sites=Index[])

User-facing constructor for [`LeftRightOperator`](@ref).

Both MPOs must use the same physical sites. Leave the process-tensor sites empty
for lazy binding.

# Examples
```julia
A = MPO(O_A, sites)
B = MPO(O_B, sites)
add!(seq, left_right_operator(A, B), 2)
```
"""
function left_right_operator(
    left::AbstractMPO{Hilbert},
    right::AbstractMPO{Hilbert},
    input_pt_sites::AbstractVector{<:Index}=Index[],
    output_pt_sites::AbstractVector{<:Index}=Index[],
)
    input_vec, output_vec = _bind_two_leg_sites(
        "LeftRightOperator", input_pt_sites, output_pt_sites; lazy_ok=true,
    )
    return LeftRightOperator(left, right, input_vec, output_vec)
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

"""
    left_action(A::AbstractMPO{Hilbert}) -> LeftRightOperator
    left_action(O::OpSum, phys_sites) -> LeftRightOperator

Build the left-action superoperator ``\\rho \\mapsto A\\rho``.

# Examples
```julia
O = OpSum()
O += 1.0, "Sz", 1
add!(seq, left_action(O, sites), 2)
```
"""
function left_action(A::AbstractMPO{Hilbert})
    phys_sites = _phys_sites_from_hilbert_mpo(A)
    os = OpSum()
    for j in eachindex(phys_sites)
        os += 1.0, "Id", j
    end
    return LeftRightOperator(A, MPO(os, phys_sites), Index[], Index[])
end

"""
    left_action(O::OpSum, phys_sites) -> LeftRightOperator

Build the left-action superoperator from an observable `OpSum` and explicit
physical Hilbert-space sites.
"""
function left_action(O::OpSum, phys_sites::AbstractVector{<:Index})
    return left_action(MPO(O, phys_sites))
end

function left_action(composed::_ComposedSingleLegInstrument, phys_sites::AbstractVector{<:Index})
    return left_action(_fold_observable_factors(composed.factors, phys_sites))
end

"""
    right_action(B::AbstractMPO{Hilbert}) -> LeftRightOperator
    right_action(O::OpSum, phys_sites) -> LeftRightOperator

Build the right-action superoperator ``\\rho \\mapsto \\rho B``.

# Examples
```julia
O = OpSum()
O += 1.0, "Sz", 1
add!(seq, right_action(O, sites), 3)
```
"""
function right_action(B::AbstractMPO{Hilbert})
    phys_sites = _phys_sites_from_hilbert_mpo(B)
    os = OpSum()
    for j in eachindex(phys_sites)
        os += 1.0, "Id", j
    end
    return LeftRightOperator(MPO(os, phys_sites), B, Index[], Index[])
end

"""
    right_action(O::OpSum, phys_sites) -> LeftRightOperator

Build the right-action superoperator from an observable `OpSum` and explicit
physical Hilbert-space sites.
"""
function right_action(O::OpSum, phys_sites::AbstractVector{<:Index})
    return right_action(MPO(O, phys_sites))
end

function right_action(composed::_ComposedSingleLegInstrument, phys_sites::AbstractVector{<:Index})
    return right_action(_fold_observable_factors(composed.factors, phys_sites))
end

"""
    TraceOut

Single-leg instrument that closes a Liouville process-tensor leg with `vec(I)`.

Canonical construction uses already-normalized fields:

```julia
TraceOut(pt_sites::Vector{Index}, leg_plev::Int)
```

Prefer [`trace_out`](@ref) for ordinary use. Zero-argument `TraceOut()` is not
supported; call `trace_out()` instead.
"""
struct TraceOut <: SingleLegInstrument
    pt_sites::Vector{Index}
    leg_plev::Int

    function TraceOut(pt_sites::Vector{Index}, leg_plev::Int)
        leg_plev == _OUTPUT_PLEV || throw(
            ArgumentError("TraceOut is fixed to leg_plev=0 (output leg); got leg_plev=$leg_plev."),
        )
        isempty(pt_sites) || _validate_single_leg_sites("TraceOut", pt_sites, _OUTPUT_PLEV)
        return new(pt_sites, _OUTPUT_PLEV)
    end
end

"""
    trace_out(pt_sites=Index[]; leg_plev=0)

User-facing constructor for [`TraceOut`](@ref).

The target site must be a Liouville index of dimension ``d^2``.

# Examples
```julia
add!(seq, trace_out(), pt.nsteps)
```
"""
function trace_out(
    pt_sites::AbstractVector{<:Index}=Index[];
    leg_plev::Int=_OUTPUT_PLEV,
)
    pt_sites_vec = _bind_single_leg_sites("TraceOut", pt_sites, _OUTPUT_PLEV)
    return TraceOut(pt_sites_vec, leg_plev)
end

"""
    UnitaryPropagation

Two-leg instrument for an explicitly inserted unitary control map.

Canonical construction uses already-normalized fields:

```julia
UnitaryPropagation(input_pt_sites, output_pt_sites, H, sites)
```

Prefer [`unitary_propagation`](@ref) for system extraction and lazy PT legs.
"""
struct UnitaryPropagation{H} <: TwoLegInstrument
    input_pt_sites::Vector{Index}
    output_pt_sites::Vector{Index}
    H::H
    sites::Vector{Index}

    function UnitaryPropagation(
        input_pt_sites::Vector{Index},
        output_pt_sites::Vector{Index},
        H,
        sites::Vector{Index},
    )
        if !isempty(input_pt_sites) || !isempty(output_pt_sites)
            _validate_two_leg_map("UnitaryPropagation", input_pt_sites, output_pt_sites)
        end
        isempty(sites) && throw(ArgumentError("UnitaryPropagation: sites cannot be empty."))
        return new{typeof(H)}(input_pt_sites, output_pt_sites, H, sites)
    end
end

"""
    unitary_propagation(H::OpSum, sites)
    unitary_propagation(H_of_t::Function, sites)
    unitary_propagation(system::AbstractSystem)
    unitary_propagation(input_pt_sites, output_pt_sites, H, sites)
    unitary_propagation(input_pt_sites, output_pt_sites, system)

User-facing constructor for [`UnitaryPropagation`](@ref).

The process tensor already contains the system's baseline propagation. Use
this instrument only for explicit control operations inserted into an
instrument schedule. A time-dependent `H_of_t` is evaluated at the midpoint
`(k - 1/2) * dt` for the map connecting output time `k - 1` to input time `k`.

# Examples
```julia
seq = default_schedule(pt)
add!(seq, unitary_propagation(H_drive, system.sites), 2)
```
"""
function unitary_propagation(
    input_pt_sites::AbstractVector{<:Index},
    output_pt_sites::AbstractVector{<:Index},
    H,
    sites::AbstractVector{<:Index},
)
    input_vec, output_vec = _bind_two_leg_sites(
        "UnitaryPropagation", input_pt_sites, output_pt_sites; lazy_ok=true,
    )
    return UnitaryPropagation(input_vec, output_vec, H, Index[sites...])
end

unitary_propagation(H::OpSum, sites::AbstractVector{<:Index}) =
    unitary_propagation(Index[], Index[], H, sites)

unitary_propagation(H_of_t::Function, sites::AbstractVector{<:Index}) =
    unitary_propagation(Index[], Index[], H_of_t, sites)

unitary_propagation(
    input_pt_sites::AbstractVector{<:Index},
    output_pt_sites::AbstractVector{<:Index},
    system::AbstractSystem,
) = unitary_propagation(input_pt_sites, output_pt_sites, system.H, system.sites)

unitary_propagation(system::AbstractSystem) =
    unitary_propagation(system.H, system.sites)

"""
    IdentityOperation

Two-leg identity connector between adjacent process-tensor legs.

Canonical construction uses already-normalized fields:

```julia
IdentityOperation(input_pt_sites::Vector{Index}, output_pt_sites::Vector{Index})
```

Prefer [`identity_operation`](@ref) for ordinary use. Zero-argument
`IdentityOperation()` is not supported; call `identity_operation()` instead.
"""
struct IdentityOperation <: TwoLegInstrument
    input_pt_sites::Vector{Index}
    output_pt_sites::Vector{Index}

    function IdentityOperation(
        input_pt_sites::Vector{Index},
        output_pt_sites::Vector{Index},
    )
        if !isempty(input_pt_sites) || !isempty(output_pt_sites)
            _validate_two_leg_map("IdentityOperation", input_pt_sites, output_pt_sites)
        end
        return new(input_pt_sites, output_pt_sites)
    end
end

"""
    identity_operation(input_pt_sites=Index[], output_pt_sites=Index[])

User-facing constructor for [`IdentityOperation`](@ref).

# Examples
```julia
seq = InstrumentSeq(default=identity_operation(), nsteps=pt.nsteps)
```
"""
function identity_operation(
    input_pt_sites::AbstractVector{<:Index}=Index[],
    output_pt_sites::AbstractVector{<:Index}=Index[],
)
    input_vec, output_vec = _bind_two_leg_sites(
        "IdentityOperation", input_pt_sites, output_pt_sites; lazy_ok=true,
    )
    return IdentityOperation(input_vec, output_vec)
end

"""
    OpenOutput

Single-leg bookkeeping instrument that leaves one output process-tensor leg
uncontracted.

Canonical construction uses already-normalized fields:

```julia
OpenOutput(pt_sites::Vector{Index}, leg_plev::Int)
```

Prefer [`open_output`](@ref) for ordinary use. Zero-argument `OpenOutput()` is
not supported; call `open_output()` instead.
"""
struct OpenOutput <: SingleLegInstrument
    pt_sites::Vector{Index}
    leg_plev::Int

    function OpenOutput(pt_sites::Vector{Index}, leg_plev::Int)
        leg_plev == _OUTPUT_PLEV || throw(
            ArgumentError("OpenOutput is fixed to leg_plev=0 (output leg); got leg_plev=$leg_plev."),
        )
        isempty(pt_sites) || _validate_single_leg_sites("OpenOutput", pt_sites, _OUTPUT_PLEV)
        return new(pt_sites, _OUTPUT_PLEV)
    end
end

"""
    open_output(pt_sites=Index[]; leg_plev=0)

User-facing constructor for [`OpenOutput`](@ref).

Place at the terminal slot `pt.nsteps` to return the final reduced state.
`open_output` materializes as the scalar no-op `ITensor(1.0)`.

# Examples
```julia
add!(seq, open_output(), pt.nsteps)
```
"""
function open_output(
    pt_sites::AbstractVector{<:Index}=Index[];
    leg_plev::Int=_OUTPUT_PLEV,
)
    pt_sites_vec = _bind_single_leg_sites("OpenOutput", pt_sites, _OUTPUT_PLEV)
    return OpenOutput(pt_sites_vec, leg_plev)
end

"""
    OpenInput

Single-leg bookkeeping instrument that leaves one input process-tensor leg
uncontracted.

Canonical construction uses already-normalized fields:

```julia
OpenInput(pt_sites::Vector{Index}, leg_plev::Int)
```

Prefer [`open_input`](@ref) for ordinary use. Zero-argument `OpenInput()` is not
supported; call `open_input()` instead.
"""
struct OpenInput <: SingleLegInstrument
    pt_sites::Vector{Index}
    leg_plev::Int

    function OpenInput(pt_sites::Vector{Index}, leg_plev::Int)
        leg_plev == _INPUT_PLEV || throw(
            ArgumentError("OpenInput is fixed to leg_plev=1 (input leg); got leg_plev=$leg_plev."),
        )
        isempty(pt_sites) || _validate_single_leg_sites("OpenInput", pt_sites, _INPUT_PLEV)
        return new(pt_sites, _INPUT_PLEV)
    end
end

"""
    open_input(pt_sites=Index[]; leg_plev=1)

User-facing constructor for [`OpenInput`](@ref).

Not valid at the terminal slot `pt.nsteps` (no input leg exists there).
"""
function open_input(
    pt_sites::AbstractVector{<:Index}=Index[];
    leg_plev::Int=_INPUT_PLEV,
)
    pt_sites_vec = _bind_single_leg_sites("OpenInput", pt_sites, _INPUT_PLEV)
    return OpenInput(pt_sites_vec, leg_plev)
end

"""
    OpenInOut

Two-leg bookkeeping instrument that leaves both input and output process-tensor
legs at one evolve slot uncontracted.

Canonical construction uses already-normalized fields:

```julia
OpenInOut(input_pt_sites::Vector{Index}, output_pt_sites::Vector{Index})
```

Prefer [`open_inout`](@ref) for ordinary use. Zero-argument `OpenInOut()` is not
supported; call `open_inout()` instead.
"""
struct OpenInOut <: TwoLegInstrument
    input_pt_sites::Vector{Index}
    output_pt_sites::Vector{Index}

    function OpenInOut(
        input_pt_sites::Vector{Index},
        output_pt_sites::Vector{Index},
    )
        if !isempty(input_pt_sites) || !isempty(output_pt_sites)
            _validate_two_leg_map("OpenInOut", input_pt_sites, output_pt_sites)
        end
        return new(input_pt_sites, output_pt_sites)
    end
end

"""
    open_inout(input_pt_sites=Index[], output_pt_sites=Index[])

User-facing constructor for [`OpenInOut`](@ref).

Unlike [`identity_operation`](@ref), which inserts `delta(in, out)`,
`open_inout` materializes as `ITensor(1.0)` and does not connect the legs.
"""
function open_inout(
    input_pt_sites::AbstractVector{<:Index}=Index[],
    output_pt_sites::AbstractVector{<:Index}=Index[],
)
    input_vec, output_vec = _bind_two_leg_sites(
        "OpenInOut", input_pt_sites, output_pt_sites; lazy_ok=true,
    )
    return OpenInOut(input_vec, output_vec)
end

"""
    ProductInstrument

Two-leg instrument at one evolve slot: an output-leg factor (`plev = 0`, time `step - 1`)
and an input-leg factor (`plev = 1`, time `step`). Construct by multiplying any two
[`SingleLegInstrument`](@ref) factors with `*` when one is on the output leg and one on
the input leg (order-independent).

# Examples
```julia
O_A, O_B = OpSum(), OpSum()
O_A += 1.0, "Sz", 1
O_B += 1.0, "Sx", 1
prod_instr = observable_measurement(O_B) * observable_measurement(O_A; leg_plev=1)
add!(seq, prod_instr, 2)
```
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

Canonical construction uses already-normalized fields:

```julia
CustomTwoLegInstrument(data, input_pt_sites, output_pt_sites, source_input, source_output)
```

Prefer [`custom_twoleg_instrument`](@ref) for keyword and positional convenience
forms.
"""
struct CustomTwoLegInstrument <: TwoLegInstrument
    data::ITensor
    input_pt_sites::Vector{Index}
    output_pt_sites::Vector{Index}
    source_input::Vector{Index}
    source_output::Vector{Index}

    function CustomTwoLegInstrument(
        data::ITensor,
        input_pt_sites::Vector{Index},
        output_pt_sites::Vector{Index},
        source_input::Vector{Index},
        source_output::Vector{Index},
    )
        if isempty(source_input) && isempty(source_output)
            _validate_two_leg_map("CustomTwoLegInstrument", input_pt_sites, output_pt_sites)
            for s in input_pt_sites
                hasind(data, s) || throw(
                    ArgumentError("CustomTwoLegInstrument: data is missing input index $s."),
                )
            end
            for s in output_pt_sites
                hasind(data, s) || throw(
                    ArgumentError("CustomTwoLegInstrument: data is missing output index $s."),
                )
            end
            expected = length(input_pt_sites) + length(output_pt_sites)
            length(inds(data)) == expected || throw(
                ArgumentError(
                    "CustomTwoLegInstrument: data must have exactly $(expected) indices; got $(length(inds(data))).",
                ),
            )
            return new(data, input_pt_sites, output_pt_sites, Index[], Index[])
        end

        if !isempty(input_pt_sites) && length(source_input) != length(input_pt_sites)
            throw(
                ArgumentError(
                    "CustomTwoLegInstrument: source_input and input_pt_sites must have equal length; " *
                    "got $(length(source_input)) and $(length(input_pt_sites)).",
                ),
            )
        end
        if !isempty(output_pt_sites) && length(source_output) != length(output_pt_sites)
            throw(
                ArgumentError(
                    "CustomTwoLegInstrument: source_output and output_pt_sites must have equal length; " *
                    "got $(length(source_output)) and $(length(output_pt_sites)).",
                ),
            )
        end
        if !isempty(input_pt_sites) || !isempty(output_pt_sites)
            _validate_two_leg_map("CustomTwoLegInstrument", input_pt_sites, output_pt_sites)
        end
        for s in source_input
            hasind(data, s) || throw(
                ArgumentError("CustomTwoLegInstrument: data is missing input index $s."),
            )
        end
        for s in source_output
            hasind(data, s) || throw(
                ArgumentError("CustomTwoLegInstrument: data is missing output index $s."),
            )
        end
        expected = length(source_input) + length(source_output)
        length(inds(data)) == expected || throw(
            ArgumentError(
                "CustomTwoLegInstrument: data must have exactly $(expected) indices; got $(length(inds(data))).",
            ),
        )
        return new(data, input_pt_sites, output_pt_sites, source_input, source_output)
    end
end

"""
    custom_twoleg_instrument(; data, input_pt_sites=Index[], output_pt_sites=Index[],
                             source_input=Index[], source_output=Index[])
    custom_twoleg_instrument(data, input_pt_sites, output_pt_sites)
    custom_twoleg_instrument(data; source_input, source_output, input_pt_sites=Index[],
                             output_pt_sites=Index[])

User-facing constructor for [`CustomTwoLegInstrument`](@ref).

Use `source_input` and `source_output` when `data` must be reindexed onto the
target process-tensor legs at contraction time.

# Examples
```julia
U = liouvillian_propagator(H, s_L, dt; jump_ops)
instr = custom_twoleg_instrument(; data=U, input_pt_sites=[in_k], output_pt_sites=[out_k])
add!(seq, instr, 1)
```
"""
function custom_twoleg_instrument(;
    data::ITensor,
    input_pt_sites::AbstractVector{<:Index}=Index[],
    output_pt_sites::AbstractVector{<:Index}=Index[],
    source_input::AbstractVector{<:Index}=Index[],
    source_output::AbstractVector{<:Index}=Index[],
)
    return CustomTwoLegInstrument(
        data,
        Index[input_pt_sites...],
        Index[output_pt_sites...],
        Index[source_input...],
        Index[source_output...],
    )
end

custom_twoleg_instrument(
    data::ITensor,
    input_pt_sites::AbstractVector{<:Index},
    output_pt_sites::AbstractVector{<:Index},
) = custom_twoleg_instrument(; data, input_pt_sites, output_pt_sites)

function custom_twoleg_instrument(
    data::ITensor;
    source_input::AbstractVector{<:Index},
    source_output::AbstractVector{<:Index},
    input_pt_sites::AbstractVector{<:Index}=Index[],
    output_pt_sites::AbstractVector{<:Index}=Index[],
)
    return custom_twoleg_instrument(;
        data,
        source_input,
        source_output,
        input_pt_sites,
        output_pt_sites,
    )
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
        return ProductInstrument(b, a)
    elseif a.leg_plev == _INPUT_PLEV && b.leg_plev == _OUTPUT_PLEV
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

"""
    InstrumentSeq

Schedule of instruments to contract with a [`ProcessTensor`](@ref ProcessTensors.ProcessTensor).
"""
mutable struct InstrumentSeq
    default::AbstractInstrument
    entries::Dict{Int,AbstractInstrument}
    nsteps::Int # upper bound for validation; 0 = unchecked until bound to a ProcessTensor
end

"""
    InstrumentSeq(default, nsteps=0; init=nothing, overrides=Dict(), entries=nothing)
    InstrumentSeq(; default, nsteps=0, entries=Dict())

Schedule of instruments to contract with a process tensor.

`default` fills unspecified evolve slots. `entries[0]` is reserved for the
initial [`StatePreparation`](@ref).

# Examples
```julia
seq = InstrumentSeq(default=identity_operation(), nsteps=pt.nsteps)
add!(seq, state_preparation(ρ0), 0)
add!(seq, trace_out(), pt.nsteps)
```
"""
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
    InstrumentSeq(; default, nsteps=0, entries=Dict())

Construct an instrument schedule using keyword arguments.
"""
function InstrumentSeq(; default::AbstractInstrument, nsteps::Int=0, entries=Dict{Int,AbstractInstrument}())
    return InstrumentSeq(default, nsteps; entries=entries)
end

"""
    resolve_instrument(seq, k)
    resolve_instrument(seq, k, fallback)

Return the instrument active at logical timestep `k` in an
[`InstrumentSeq`](@ref).

For ordinary evolve slots (`k ≥ 1`), an explicit `seq.entries[k]` wins;
otherwise the schedule default is used. With a `fallback` argument, that
instrument is used instead of `seq.default` when no explicit entry is present.

At `k = 0` (initial preparation), return the explicit entry if present, else
`nothing` — there is no default preparation.

This is part of the public `ProcessTensors.Instruments` API. Prefer
[`add!`](@ref) / [`evaluate_process`](@ref ProcessTensors.evaluate_process) for
ordinary schedule workflows; use `resolve_instrument` when inspecting or
manually materialising a schedule slot.

# Examples
```julia
using ProcessTensors.Instruments: resolve_instrument

seq = InstrumentSeq(default=identity_operation(), nsteps=pt.nsteps)
add!(seq, state_preparation(ρ0), 0)
add!(seq, trace_out(), pt.nsteps)

@assert resolve_instrument(seq, 0) isa StatePreparation
@assert resolve_instrument(seq, 1) isa IdentityOperation
@assert resolve_instrument(seq, pt.nsteps) isa TraceOut
```
"""
function resolve_instrument(seq::InstrumentSeq, k::Int)
    k == 0 && return get(seq.entries, 0, nothing)
    k >= 1 || throw(ArgumentError("resolve_instrument: expected k ≥ 0; got $k."))
    return get(seq.entries, k, seq.default)
end

"""
    resolve_instrument(seq, k, fallback)

See [`resolve_instrument`](@ref resolve_instrument(::InstrumentSeq, ::Int)).
"""
function resolve_instrument(seq::InstrumentSeq, k::Int, fallback::AbstractInstrument)
    k == 0 && return get(seq.entries, 0, nothing)
    k >= 1 || throw(ArgumentError("resolve_instrument: expected k ≥ 0; got $k."))
    return get(seq.entries, k, fallback)
end

"""
    add!(seq, instr, tstep)

Insert `instr` at `tstep`, replacing an existing explicit entry at that timestep.

Only `StatePreparation` is allowed at `tstep = 0`.
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
    _instrument_leg_maps(seq, nsteps)

Return dictionaries describing which instruments cover process-tensor input and
output time labels, together with any missing input/output labels.

Open instruments still claim legs even when they materialize as `ITensor(1.0)`.
"""
function _instrument_leg_maps(seq::InstrumentSeq, nsteps::Int)
    nsteps >= 1 || throw(ArgumentError("_instrument_leg_maps: nsteps must be >= 1"))

    in_map = Dict{Int,AbstractInstrument}()
    out_map = Dict{Int,AbstractInstrument}()
    consumed = falses(nsteps + 1)

    function _record!(instr::AbstractInstrument, step::Int)
        if instr isa TwoLegInstrument
            step <= nsteps - 1 && (in_map[step] = instr)
            tout = step - 1
            tout <= nsteps - 2 && (out_map[tout] = instr)
        elseif instr isa SingleLegInstrument
            if instr.leg_plev == _OUTPUT_PLEV
                tout = step - 1
                tout <= nsteps - 2 && (out_map[tout] = instr)
            else
                step <= nsteps - 1 && (in_map[step] = instr)
            end
        else
            throw(
                ArgumentError(
                    "_instrument_leg_maps: unsupported instrument $(typeof(instr)) at tstep=$step.",
                ),
            )
        end
        return nothing
    end

    # Physical single-leg pairs (same interpretation as create_instruments).
    for s in 1:(nsteps - 1)
        out_entry = resolve_instrument(seq, s)
        in_entry = resolve_instrument(seq, s + 1)
        out_entry isa SingleLegInstrument || continue
        in_entry isa SingleLegInstrument || continue
        out_entry.leg_plev == _OUTPUT_PLEV || continue
        in_entry.leg_plev == _INPUT_PLEV || continue
        out_entry isa OpenOutput && continue
        _record!(ProductInstrument(in_entry, out_entry), s)
        consumed[s + 1] = true
    end

    for step in 1:nsteps
        consumed[step] && continue
        _record!(resolve_instrument(seq, step), step)
    end
    for step in 1:nsteps
        consumed[step] || continue
        _record!(seq.default, step)
    end

    prep = resolve_instrument(seq, 0)
    if prep !== nothing
        prep isa StatePreparation ||
            (prep isa _ComposedSingleLegInstrument && any(f -> f isa StatePreparation, prep.factors)) ||
            throw(ArgumentError("_instrument_leg_maps: tstep=0 must be StatePreparation"))
        in_map[0] = prep
    end

    expected_in = collect(0:nsteps-1)
    expected_out = nsteps == 1 ? Int[] : collect(0:nsteps-2)
    missing_in = [k for k in expected_in if !haskey(in_map, k)]
    missing_out = [k for k in expected_out if !haskey(out_map, k)]

    return in_map, out_map, missing_in, missing_out
end
