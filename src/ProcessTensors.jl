# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: src/ProcessTensors.jl
# Contributor: Gauthameshwar S.
#
# Defines the main ProcessTensors.jl module, package imports, includes, exports,
# and public API surface.

module ProcessTensors

using ITensors
import ITensorMPS

# Space model and tensor wrappers

include("basis.jl")
using .Basis: AbstractSpace, Hilbert, Liouville

include("mps/mps.jl")
include("mpo/mpo.jl")

# Tensor-network operations

include("networks/indices.jl")
include("networks/algebra.jl")
include("networks/observables.jl")

# MPS and MPO construction

include("mps/constructors.jl")
include("mpo/constructors.jl")

# Operators, Liouvillians, and time evolution

include("hamiltonian.jl")
include("liouvillian.jl")
include("time_evolution/tdvp.jl")
include("time_evolution/trotter.jl")
include("time_evolution/tebd.jl")

# Systems and environments

include("systems/systems.jl")

include("environments/spectrals.jl")
using .Spectrals: AbstractSpectralDensity

include("environments/environments.jl")
using .Environments: AbstractBathMode, AbstractBath
using .Environments: BosonicMode, SpinMode
using .Environments: BosonicBath, SpinBath
using .Environments: bosonic_mode, spin_mode
using .Environments: bosonic_bath, spin_bath
using .Environments: mode_initial_states

# Instruments

include("instruments/Instruments.jl")
using .Instruments: AbstractInstrument, SingleLegInstrument, TwoLegInstrument
using .Instruments: StatePreparation, ObservableMeasurement, TraceOut
using .Instruments: IdentityOperation, UnitaryPropagation, LeftRightOperator
using .Instruments: OpenOutput, OpenInput, OpenInOut
using .Instruments: ProductInstrument, CustomTwoLegInstrument
using .Instruments: state_preparation, observable_measurement, trace_out
using .Instruments: identity_operation, unitary_propagation
using .Instruments: left_right_operator
using .Instruments: open_output, open_input, open_inout
using .Instruments: custom_twoleg_instrument
using .Instruments: left_action, right_action
using .Instruments: InstrumentSeq
using .Instruments: add!

# Process tensors

include("process_tensor/process_tensor.jl")

# Instrument materialisation depends on process-tensor leg definitions
Base.include(
    Instruments,
    joinpath(@__DIR__, "instruments/itensor_instruments.jl"),
)

include("builders/abstract_builders.jl")
include("builders/dense_process_tensor.jl")
include("process_tensor/build.jl")
include("process_tensor/evaluate.jl")
include("process_tensor/evolve.jl")
include("process_tensor/multitime.jl")

# Public API

# Space-aware tensor-network objects
export Hilbert, Liouville
export AbstractMPS, AbstractMPO
export MPS, MPO
export random_mps, random_mpo
export outer, projector

# Tensor-network inspection
export siteinds, siteind
export linkinds, linkind, linkdim
export linkdims, maxlinkdim

# Tensor-network algebra and observables
export apply, contract, add
export inner, dot, norm
export expect, correlation_matrix, entropy, tr

# Index-tag helpers
export tag_tokens, has_tag_token, has_tag_prefix, tag_value

# Hilbert/Liouville conversion
export liouv_sites
export to_dm, to_liouville, to_hilbert

# Operators and Liouvillians
export OpSum, add!, op
export liouvillian_opsum, liouvillian_mpo
export liouvillian_propagator

# Deprecated compatibility aliases; remove after the migration window.
export OpSum_Liouville, MPO_Liouville, liouvillian_propagator_itensor

# Systems
export AbstractSystem
export SpinSystem, BosonSystem
export spin_system, boson_system

# Environments
export AbstractBathMode, AbstractBath
export BosonicMode, SpinMode
export BosonicBath, SpinBath
export bosonic_mode, spin_mode
export bosonic_bath, spin_bath
export AbstractSpectralDensity
export mode_initial_states

# Instruments
export AbstractInstrument, SingleLegInstrument, TwoLegInstrument
export InstrumentSeq

export StatePreparation, ObservableMeasurement, TraceOut
export IdentityOperation, UnitaryPropagation, LeftRightOperator
export OpenOutput, OpenInput, OpenInOut
export ProductInstrument, CustomTwoLegInstrument

export state_preparation, observable_measurement, trace_out
export identity_operation, unitary_propagation
export left_right_operator
export open_output, open_input, open_inout
export custom_twoleg_instrument

export left_action, right_action

# Process tensors
export Dense
export ProcessTensor
export build_process_tensor, default_schedule
export evaluate_process, evolve
export two_time_correlation_seq
export isfullycontracted, open_leg_info
export input_sites, output_sites
export coupling_times, coupling_sites

# Space marker
export space

# Time evolution
export tdvp, tebd

end # module ProcessTensors
