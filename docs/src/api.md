```@meta
CurrentModule = ProcessTensors
```

# API Reference

## Hilbert/Liouville wrapper model

`Hilbert` and `Liouville` are ProcessTensors space labels for wrapped
`ITensorMPS` objects. `MPS{Hilbert}` and `MPO{Hilbert}` represent ordinary
states and operators, while `MPS{Liouville}` and `MPO{Liouville}` represent
vectorized density operators and superoperators.

The wrapped `ITensorMPS` object lives in `.core`. Most generic ITensorMPS
operations act on `.core` and rewrap the result with the same `Hilbert` or
`Liouville` label. Liouville wrappers may also store `combiners`, which record
how Hilbert bra/ket site pairs were fused so that `to_hilbert` can reconstruct
the original density MPO.

Use `liouv_sites` to create Liouville site indices. Reuse the exact same
`Index` objects across `to_liouville`, `liouvillian_mpo`, systems, baths, and
process-tensor instruments so ITensor contractions match by index identity.
Process-tensor input legs are primed (`plev = 1`) and output legs are unprimed
(`plev = 0`).


## List of ITensorMPS functions exported by ProcessTensors

ProcessTensors exposes a compact root surface for ordinary Hilbert/Liouville
workflows. These names are the shared ITensorMPS generics, extended with methods
that act on wrappers and rewrap results when appropriate, preserving the
`Hilbert` or `Liouville` space marker. 
Here is a list of `ITensorMPS.jl` functions that is available in `ProcessTensors.jl`: 

```julia
siteinds, siteind, linkinds, linkind, linkdim, linkdims, maxlinkdim

apply, contract, add

random_mps, random_mpo, outer, projector

inner, dot, norm, expect, correlation_matrix, entropy, tr

OpSum, add!, op
```

For generic MPS/MPO algorithmic details and keyword arguments, refer to the
ITensorMPS documentation.

### Advanced tensor-network surgery

Canonical forms, truncation diagnostics, bond moves, sampling, and related
technical operations are not defined by ProcessTensors. Use ITensorMPS
directly, typically on native cores for such algorithms. for example,

```julia
import ITensorMPS
expanded_core = ITensorMPS.expand(state.core, operator.core; alg="global_krylov", ...)
ITensorMPS.orthogonalize!(expanded_core, 1)
```

## API Documentation

### Space tags

```@docs
Hilbert
Liouville
```

### MPS and MPO wrappers

```@docs
AbstractMPS
MPS
```

```@docs
AbstractMPO
MPO
```

### MPS/MPO constructors

```@docs
random_mps
random_mpo
```

### Hilbert/Liouville conversion

```@docs
liouv_sites
```

```@docs
to_dm(::AbstractMPS{Hilbert})
to_dm(::AbstractVector{<:AbstractMPS{Hilbert}})
```

```@docs
to_liouville(::AbstractMPO{Hilbert})
to_liouville(::AbstractMPS{Hilbert})
```

```@docs
to_hilbert
```

### Liouvillian builders

!!! compat "Renamed Liouvillian constructors"
    `OpSum_Liouville`, `MPO_Liouville`, and
    `liouvillian_propagator_itensor` are deprecated aliases of
    `liouvillian_opsum`, `liouvillian_mpo`, and `liouvillian_propagator`.
    Prefer the new names; the aliases will be removed in a later `0.3+`
    release after the migration window.

```@docs
liouvillian_opsum(::OpSum)
liouvillian_opsum(::OpSum, ::Tuple{<:Number,<:AbstractString,<:Integer})
liouvillian_opsum(::OpSum, ::AbstractVector{<:Tuple{<:Number,<:AbstractString,<:Integer}})
liouvillian_opsum(::OpSum, ::OpSum)
liouvillian_opsum(::OpSum, ::AbstractVector{<:OpSum})
```

```@docs
liouvillian_mpo(::OpSum, ::AbstractVector{<:Index})
liouvillian_mpo(::OpSum, ::Any, ::AbstractVector{<:Index})
```

```@docs
liouvillian_propagator
```

### Process tensors

```@docs
ProcessTensor
Dense
```

```@docs
input_sites
output_sites
coupling_times
coupling_sites
default_schedule
isfullycontracted
open_leg_info
```

```@docs
build_process_tensor(::AbstractSystem, ::Index)
build_process_tensor(::AbstractSystem)
```

```@docs
evaluate_process(::ProcessTensor, ::InstrumentSeq)
evaluate_process(::ProcessTensor, ::AbstractVector{<:InstrumentSeq})
evaluate_process(::ProcessTensor, ::Any, ::InstrumentSeq)
evaluate_process(::ProcessTensor, ::Any)
```

```@docs
evolve(::ProcessTensor, ::InstrumentSeq)
evolve(::ProcessTensor, ::Any, ::InstrumentSeq)
evolve(::ProcessTensor, ::Any)
```

```@docs
two_time_correlation_seq
```

### Systems

```@docs
AbstractSystem
SpinSystem
BosonSystem
```

```@docs
spin_system
boson_system
```

### Baths and spectral densities

```@docs
AbstractBathMode
BosonicMode
SpinMode
AbstractBath
BosonicBath
SpinBath
```

```@docs
bosonic_mode
spin_mode
bosonic_bath
spin_bath
mode_initial_states
```

```@docs
ProcessTensors.Spectrals.AbstractSpectralDensity
ProcessTensors.Spectrals.OhmicSpectralDensity
ProcessTensors.Spectrals.LorentzianSpectralDensity
ProcessTensors.Spectrals.ohmic_sd
ProcessTensors.Spectrals.lorentzian_sd
```

### Instruments and schedules

```@docs
AbstractInstrument
SingleLegInstrument
TwoLegInstrument
```

```@docs
StatePreparation
ObservableMeasurement
TraceOut
LeftRightOperator
UnitaryPropagation
IdentityOperation
OpenOutput
OpenInput
OpenInOut
CustomTwoLegInstrument
ProductInstrument
```

```@docs
state_preparation
observable_measurement
trace_out
left_right_operator
unitary_propagation
identity_operation
open_output
open_input
open_inout
custom_twoleg_instrument
```

```@docs
left_action(::AbstractMPO{Hilbert})
left_action(::OpSum, ::AbstractVector{<:Index})
right_action(::AbstractMPO{Hilbert})
right_action(::OpSum, ::AbstractVector{<:Index})
```

```@docs
InstrumentSeq
add!
```

Schedule inspection and advanced dense materialisation live in the Instruments
submodule. Ordinary workflows keep schedules lazy and call `evaluate_process` /
`evolve`; the names below are for explicit materialisation and diagnostics:

```@docs
ProcessTensors.Instruments.resolve_instrument
ProcessTensors.Instruments.instrument_leg_maps
ProcessTensors.Instruments.instrument_itensor
ProcessTensors.Instruments.create_instruments
```

### Time evolution

```@docs
tebd(::AbstractMPS{Hilbert}, ::OpSum, ::Real, ::Real)
tebd(::AbstractMPS{Liouville}, ::OpSum, ::Real, ::Real)
tdvp
```

Algorithm selectors come from upstream ITensors:

```julia
using ITensors.Ops: Exact, Trotter

tebd(psi, H, dt, T; alg=Trotter{2}())
liouvillian_propagator(H, sites_L, dt; alg=Exact())
```

Advanced gate construction is available by qualification (not root-exported):

```julia
gates = ProcessTensors.trotter_gates(H, sites, -im * dt; alg=Trotter{2}())
U, final_out = ProcessTensors.propagator_itensor_from_gates(gates, sites_L)
```

```@docs
trotter_gates
propagator_itensor_from_gates
```

### Tag utilities

```@docs
tag_tokens
has_tag_token
has_tag_prefix
tag_value
```
