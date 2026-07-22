# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: test/process_tensor/test_api_names.jl
# Contributor: Gauthameshwar S.
#
# Tests public export visibility for core space types, Liouvillian constructors,
# and domain type / user-facing-constructor pairs.
#
# Run with:
#   julia --project=. test/runtests.jl

using ProcessTensors
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
