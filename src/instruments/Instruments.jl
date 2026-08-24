# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: src/instruments/Instruments.jl
# Contributor: Gauthameshwar S.
#
# Defines the Instruments submodule shell and exports for process-tensor
# instruments, memory-bearing testers, and schedules.

module Instruments

import ..ProcessTensors
import ..ProcessTensors: add!
using ITensors
using LinearAlgebra
import ITensors.Ops: Exact, Trotter
using ..ProcessTensors: AbstractMPO, AbstractMPS, AbstractSystem, Hilbert, Liouville, MPO,
                        OpSum, liouvillian_opsum, Index, ITensor, apply, dim, plev, prime,
                        replaceind, siteinds, tag_tokens, tag_value, has_tag_token,
                        liouv_sites, to_dm, to_liouville,
                        _phys_site_from_liouv, _superop_matrix, _LiouvLeft, _LiouvRight,
                        _liouv_site_type, liouvillian_propagator,
                        _phys_sites_from_hilbert_mpo, _hilbert_itensor_to_liouville

export AbstractInstrument, SingleLegInstrument, TwoLegInstrument,
       StatePreparation, ObservableMeasurement, TraceOut,
       IdentityOperation, UnitaryPropagation, OpenOutput, OpenInput, OpenInOut,
       ProductInstrument, CustomTwoLegInstrument,
       LeftRightOperator, left_action, right_action,
       state_preparation, observable_measurement, trace_out,
       left_right_operator, unitary_propagation, identity_operation,
       open_output, open_input, open_inout,
       custom_twoleg_instrument,
       InstrumentSeq, add!, resolve_instrument, instrument_leg_maps,
       instrument_itensor, create_instruments,
       Tester, tester, AbstractTesterAction,
       TesterIdentity, TesterPropagation, TesterUnitary,
       JointPropagation, JointUnitary,
       tester_identity, tester_propagation, tester_unitary,
       joint_propagation, joint_unitary,
       TesterSeq, resolve_tester_action

include("lazy_instruments.jl")
include("testers.jl")
include("tester_compile.jl")

end # module Instruments
