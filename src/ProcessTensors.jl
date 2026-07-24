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

# Foundation

include("basis.jl")
using .Basis: AbstractSpace, Hilbert, Liouville

# Space marker accessor for Hilbert/Liouville wrappers.
function space end

# Core Types (MPS & MPO structs, getproperty, show)

include("mps/mps.jl")
include("mpo/mpo.jl")

# Rewrap utility — converts a raw CoreMPS/CoreMPO back into our wrapper type
function _rewrap(m::AbstractMPS{S}, new_core) where {S <: AbstractSpace}
    if m isa MPS
        return S === Hilbert ? MPS{Hilbert}(new_core) : MPS{Liouville}(new_core, m.combiners)
    else
        return S === Hilbert ? MPO{Hilbert}(new_core) : MPO{Liouville}(new_core, m.combiners)
    end
end

# Compact tensor-network surface (ITensorMPS-shared generics + wrapper methods)

include("networks/indices.jl")
include("networks/algebra.jl")
include("networks/observables.jl")

# MPS / MPO constructors

include("mps/constructors.jl")
include("mpo/constructors.jl")

# Hamiltonian / Operator Sums

include("hamiltonian.jl")

# Liouvillian

include("liouvillian.jl")

# Time Evolution

include("time_evolution/tdvp.jl")
include("time_evolution/trotter.jl")
include("time_evolution/tebd.jl")

# Systems / Baths / Instruments

include("systems/systems.jl")
include("environments/spectrals.jl")
using .Spectrals: AbstractSpectralDensity
include("environments/environments.jl")
using .Environments: AbstractBathMode, AbstractBath, BosonicMode, SpinMode, BosonicBath, SpinBath,
                    bosonic_mode, spin_mode, bosonic_bath, spin_bath,
                    mode_initial_states
include("instruments/Instruments.jl")
using .Instruments: AbstractInstrument, SingleLegInstrument, TwoLegInstrument,
                    StatePreparation, ObservableMeasurement,
                    TraceOut, IdentityOperation, UnitaryPropagation,
                    OpenOutput, OpenInput, OpenInOut, ProductInstrument,
                    CustomTwoLegInstrument,
                    LeftRightOperator, left_action, right_action,
                    state_preparation, observable_measurement, trace_out,
                    left_right_operator, unitary_propagation, identity_operation,
                    open_output, open_input, open_inout,
                    custom_twoleg_instrument,
                    InstrumentSeq, add!

# Process Tensors

include("process_tensor/process_tensor.jl")
Base.include(Instruments, joinpath(@__DIR__, "instruments/itensor_instruments.jl"))
include("builders/abstract_builders.jl")
include("builders/dense_process_tensor.jl")
include("process_tensor/build.jl")
include("process_tensor/evaluate.jl")
include("process_tensor/evolve.jl")
include("process_tensor/multitime.jl")

# Exports (grouped by category)

# Core types
export AbstractMPS, AbstractMPO, MPS, MPO, Hilbert, Liouville
export tag_tokens, has_tag_token, has_tag_prefix, tag_value

# Compact tensor-network surface
export siteinds, siteind, linkinds, linkind, linkdim, linkdims, maxlinkdim
export apply, contract, add
export random_mps, random_mpo, outer, projector
export inner, dot, norm, expect, correlation_matrix, entropy, tr

# Operator construction
export OpSum, add!, op

# Liouvillian
export to_dm, to_liouville, to_hilbert, liouv_sites,
       liouvillian_opsum, liouvillian_mpo, liouvillian_propagator

# Deprecated compatibility exports; remove in a later 0.y release after the migration window.
export MPO_Liouville, OpSum_Liouville, liouvillian_propagator_itensor

# Wrapper space marker
export space

# Systems / Baths / Instruments / PT
export AbstractSystem, SpinSystem, BosonSystem, spin_system, boson_system

export AbstractBathMode, BosonicMode, SpinMode, bosonic_mode, spin_mode,
       AbstractBath, BosonicBath, SpinBath, bosonic_bath, spin_bath,
       mode_initial_states

export AbstractInstrument, SingleLegInstrument, TwoLegInstrument,
       StatePreparation, ObservableMeasurement, TraceOut,
       IdentityOperation, UnitaryPropagation, OpenOutput, OpenInput, OpenInOut,
       ProductInstrument, CustomTwoLegInstrument,
       LeftRightOperator, left_action, right_action,
       state_preparation, observable_measurement, trace_out,
       left_right_operator, unitary_propagation, identity_operation,
       open_output, open_input, open_inout,
       custom_twoleg_instrument,
       InstrumentSeq, add!

export Dense,
       ProcessTensor, build_process_tensor, default_schedule, evolve, evaluate_process,
       two_time_correlation_seq,
       isfullycontracted, open_leg_info,
       coupling_times, coupling_sites, input_sites, output_sites

# Time evolution
export tdvp, tebd

end # module ProcessTensors
