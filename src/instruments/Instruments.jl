# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: src/instruments/Instruments.jl
# Contributor: Gauthameshwar S.
#
# Defines the Instruments submodule shell and lazy instrument exports used by
# process-tensor schedules.

module Instruments

import ..ProcessTensors
import ..ProcessTensors: add!
using ITensors
using LinearAlgebra
using ..ProcessTensors: AbstractMPO, AbstractMPS, AbstractSystem, Hilbert, Liouville, MPO,
                        OpSum, liouvillian_opsum, Index, ITensor, apply, dim, plev, prime,
                        replaceind, siteinds, tag_value, to_dm, to_liouville,
                        _phys_site_from_liouv, _superop_matrix, _LiouvLeft, _LiouvRight,
                        liouvillian_propagator, Exact, Trotter,
                        _phys_sites_from_hilbert_mpo

export AbstractInstrument, SingleLegInstrument, TwoLegInstrument,
       StatePreparation, ObservableMeasurement, TraceOut,
       IdentityOperation, UnitaryPropagation, OpenOutput, OpenInput, OpenInOut,
       ProductInstrument, CustomTwoLegInstrument,
       LeftRightOperator, left_action, right_action,
       state_preparation, observable_measurement, trace_out,
       left_right_operator, unitary_propagation, identity_operation,
       open_output, open_input, open_inout,
       custom_twoleg_instrument,
       InstrumentSeq, add!, resolve_instrument,
       instrument_itensor, create_instruments

include("lazy_instruments.jl")

end # module Instruments
