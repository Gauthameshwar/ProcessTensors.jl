# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: test/process_tensor/test_api_names.jl
# Contributor: Gauthameshwar S.
#
# Tests public export visibility for core space types and Liouvillian
# constructors after the API rename.
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
