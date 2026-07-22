# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: test/process_tensor/test_api_names.jl
# Contributor: Gauthameshwar S.
#
# Tests public export visibility for core space types, Liouvillian constructors,
# domain type / user-facing-constructor pairs, and time-evolution / instrument
# API boundaries.
#
# Run with:
#   julia --project=. test/runtests.jl

using ProcessTensors
using ITensorMPS
using Test

@testset "API surface: core spaces and Liouvillian names" begin
    @test isdefined(ProcessTensors, :AbstractSpace)
    @test :AbstractSpace ∉ names(ProcessTensors)

    @test :Hilbert ∈ names(ProcessTensors)
    @test :Liouville ∈ names(ProcessTensors)
    @test :MPS ∈ names(ProcessTensors)
    @test :MPO ∈ names(ProcessTensors)
    @test :AbstractMPS ∈ names(ProcessTensors)
    @test :AbstractMPO ∈ names(ProcessTensors)

    @test :to_dm ∈ names(ProcessTensors)
    @test :to_liouville ∈ names(ProcessTensors)
    @test :to_hilbert ∈ names(ProcessTensors)
    @test :liouv_sites ∈ names(ProcessTensors)

    @test :liouvillian_opsum ∈ names(ProcessTensors)
    @test :liouvillian_mpo ∈ names(ProcessTensors)
    @test :liouvillian_propagator ∈ names(ProcessTensors)

    # Deprecated aliases remain defined and exported during the migration window.
    @test isdefined(ProcessTensors, :OpSum_Liouville)
    @test isdefined(ProcessTensors, :MPO_Liouville)
    @test isdefined(ProcessTensors, :liouvillian_propagator_itensor)
    @test :OpSum_Liouville ∈ names(ProcessTensors)
    @test :MPO_Liouville ∈ names(ProcessTensors)
    @test :liouvillian_propagator_itensor ∈ names(ProcessTensors)
end

@testset "API surface: domain types and user-facing constructors" begin
    pairs = (
        (:SpinSystem, :spin_system),
        (:BosonSystem, :boson_system),
        (:BosonicMode, :bosonic_mode),
        (:SpinMode, :spin_mode),
        (:BosonicBath, :bosonic_bath),
        (:SpinBath, :spin_bath),
        (:StatePreparation, :state_preparation),
        (:ObservableMeasurement, :observable_measurement),
        (:LeftRightOperator, :left_right_operator),
        (:TraceOut, :trace_out),
        (:UnitaryPropagation, :unitary_propagation),
        (:IdentityOperation, :identity_operation),
        (:OpenOutput, :open_output),
        (:OpenInput, :open_input),
        (:OpenInOut, :open_inout),
        (:CustomTwoLegInstrument, :custom_twoleg_instrument),
    )
    for (type_name, user_ctor_name) in pairs
        @test type_name ∈ names(ProcessTensors)
        @test user_ctor_name ∈ names(ProcessTensors)
    end
end

@testset "API surface: process-tensor and instrument boundaries" begin
    @test :Dense ∈ names(ProcessTensors)
    @test :isfullycontracted ∈ names(ProcessTensors)
    @test :open_leg_info ∈ names(ProcessTensors)
    @test :two_time_correlation_seq ∈ names(ProcessTensors)
    @test :AbstractPTBuilder ∉ names(ProcessTensors)
    @test isdefined(ProcessTensors, :AbstractPTBuilder)
    @test :_generate_pt_legs ∉ names(ProcessTensors)
    @test :generate_pt_legs ∉ names(ProcessTensors)
    @test :all_pt_legs_contracted ∉ names(ProcessTensors)

    @test :instrument_itensor ∉ names(ProcessTensors)
    @test :create_instruments ∉ names(ProcessTensors)
    @test :resolve_instrument ∉ names(ProcessTensors)
    @test :instrument_leg_maps ∉ names(ProcessTensors)

    @test :instrument_itensor ∈ names(ProcessTensors.Instruments)
    @test :create_instruments ∈ names(ProcessTensors.Instruments)
    @test :resolve_instrument ∈ names(ProcessTensors.Instruments)
    @test :instrument_leg_maps ∈ names(ProcessTensors.Instruments)

    @test ProcessTensors.add! === ITensorMPS.add!
    @test ProcessTensors.Instruments.add! === ProcessTensors.add!
end

@testset "API surface: time-evolution boundaries" begin
    @test :tebd ∈ names(ProcessTensors)
    @test :tdvp ∈ names(ProcessTensors)

    @test :Exact ∉ names(ProcessTensors)
    @test :Trotter ∉ names(ProcessTensors)
    @test :trotter_gates ∉ names(ProcessTensors)
    @test :propagator_itensor_from_gates ∉ names(ProcessTensors)

    @test :promote_itensor_eltype ∉ names(ProcessTensors)
    @test :convert_leaf_eltype ∉ names(ProcessTensors)
    @test :argsdict ∉ names(ProcessTensors)
    @test :sim! ∉ names(ProcessTensors)

    # Removed from the TDVP import surface (ITensorMPS utilities).
    @test !isdefined(ProcessTensors, :sim!)
    @test !isdefined(ProcessTensors, :promote_itensor_eltype)
    # `argsdict` / `convert_leaf_eltype` may still exist via `using ITensors`,
    # but ProcessTensors no longer exports them or imports them from ITensorMPS.

    @test isdefined(ProcessTensors, :trotter_gates)
    @test isdefined(ProcessTensors, :propagator_itensor_from_gates)

    @test ProcessTensors.tdvp === ITensorMPS.tdvp
end
