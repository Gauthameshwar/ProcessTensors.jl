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
`Index` objects across `to_liouville`, `MPO_Liouville`, systems, baths, and
process-tensor instruments so ITensor contractions match by index identity.
Process-tensor input legs are primed (`plev = 1`) and output legs are unprimed
(`plev = 0`).

## List of available ITensorMPS functions

ProcessTensors.jl forwards several ITensorMPS operations to the wrapped `.core`
object and rewraps results when appropriate, preserving the `Hilbert` or
`Liouville` space label.

For generic MPS/MPO algorithmic details and keyword arguments, refer to the
ITensorMPS documentation. ProcessTensors documentation focuses only on the
wrapper semantics and Hilbert/Liouville behavior.

- `siteinds`, `siteind`, `linkinds`, `linkind`, `linkdim`, `linkdims`,
  `maxlinkdim`, `common_siteind`, `common_siteinds`, `unique_siteind`,
  `unique_siteinds`, `findfirstsiteind`, `findfirstsiteinds`, `findsite`,
  `findsites`, `firstsiteind`, `firstsiteinds`, `replace_siteinds`,
  `replace_siteinds!`, `hassameinds`, `totalqn`, `replaceprime`:
  [Source Doc](https://docs.itensor.org/ITensorMPS/stable/MPSandMPO.html)
- `orthogonalize`, `orthogonalize!`, `truncate`, `truncate!`, `normalize!`,
  `isortho`, `ortho_lims`, `orthocenter`, `set_ortho_lims!`,
  `reset_ortho_lims!`: [Source Doc](https://docs.itensor.org/ITensorMPS/stable/MPSandMPO.html)
- `apply`, `contract`, `replacebond`, `replacebond!`, `swapbondsites`,
  `movesite`, `movesites`, `error_contract`:
  [Source Doc](https://docs.itensor.org/ITensorMPS/stable/MPSandMPO.html)
- `inner`, `dot`, `⋅`, `loginner`, `logdot`, `norm`, `lognorm`, `expect`,
  `correlation_matrix`, `sample`, `sample!`, `entropy`, `outer`, `projector`,
  `state`, `splitblocks`, `tr`:
  [Source Doc](https://docs.itensor.org/ITensorMPS/stable/MPSandMPO.html)
- `OpSum`, `add!`, `op`, `ops`, `coefficient`:
  [Source Doc](https://docs.itensor.org/ITensorMPS/stable/OpSum.html)
- `siteind`, `siteinds`, `state`, `op` for SiteType-based physics indices:
  [Source Doc](https://docs.itensor.org/ITensorMPS/stable/SiteType.html)

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

```@docs
OpSum_Liouville(::OpSum)
OpSum_Liouville(::OpSum, ::Tuple{<:Number,<:AbstractString,<:Integer})
OpSum_Liouville(::OpSum, ::AbstractVector{<:Tuple{<:Number,<:AbstractString,<:Integer}})
OpSum_Liouville(::OpSum, ::OpSum)
OpSum_Liouville(::OpSum, ::AbstractVector{<:OpSum})
```

```@docs
MPO_Liouville(::OpSum, ::AbstractVector{<:Index})
MPO_Liouville(::OpSum, ::Any, ::AbstractVector{<:Index})
```

```@docs
liouvillian_propagator_itensor
```

### Process tensors

```@docs
ProcessTensor
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
state_preparation
ObservableMeasurement
observable_measurement
TraceOut
trace_out
LeftRightOperator
left_right_operator
```

```@docs
left_action(::AbstractMPO{Hilbert})
left_action(::OpSum, ::AbstractVector{<:Index})
right_action(::AbstractMPO{Hilbert})
right_action(::OpSum, ::AbstractVector{<:Index})
```

```@docs
SystemPropagation
system_propagation
IdentityOperation
identity_operation
OpenOutput
open_output
ProductInstrument
CustomTwoLegInstrument
custom_twoleg_instrument
InstrumentSeq
resolve_instrument
add!
instrument_itensor
```

### Time evolution

```@docs
tebd(::AbstractMPS{Hilbert}, ::OpSum, ::Real, ::Real)
tebd(::AbstractMPS{Liouville}, ::OpSum, ::Real, ::Real)
tdvp
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
