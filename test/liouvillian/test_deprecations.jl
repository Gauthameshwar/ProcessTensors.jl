# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: test/liouvillian/test_deprecations.jl
# Contributor: Gauthameshwar S.
#
# Tests that deprecated Liouvillian constructor aliases warn and forward to the
# canonical names with numerically equivalent results.
#
# Run with:
#   julia --project=. test/runtests.jl

using ProcessTensors
using ITensors
using LinearAlgebra
using Test

@testset "API surface: deprecated Liouvillian aliases remain exported" begin
    @test isdefined(ProcessTensors, :OpSum_Liouville)
    @test isdefined(ProcessTensors, :MPO_Liouville)
    @test isdefined(ProcessTensors, :liouvillian_propagator_itensor)
    @test :OpSum_Liouville ∈ names(ProcessTensors)
    @test :MPO_Liouville ∈ names(ProcessTensors)
    @test :liouvillian_propagator_itensor ∈ names(ProcessTensors)
end

@testset "Liouvillian deprecation aliases" begin
    phys = siteinds("S=1/2", 1)
    sL = liouv_sites(phys)
    H = OpSum()
    H += 0.7, "Sz", 1
    dt = 0.05

    @testset "OpSum_Liouville forwards to liouvillian_opsum" begin
        old = @test_deprecated OpSum_Liouville(H)
        new = liouvillian_opsum(H)
        @test old isa OpSum
        @test new isa OpSum
        old_mpo = MPO(old, sL)
        new_mpo = MPO(new, sL)
        old_arr = Array(old_mpo[1], prime(sL[1]), sL[1])
        new_arr = Array(new_mpo[1], prime(sL[1]), sL[1])
        @test norm(old_arr - new_arr) < 1e-12
    end

    @testset "MPO_Liouville forwards to liouvillian_mpo" begin
        old = @test_deprecated MPO_Liouville(H, sL)
        new = liouvillian_mpo(H, sL)
        @test old isa MPO{Liouville}
        @test new isa MPO{Liouville}
        old_arr = Array(old[1], prime(sL[1]), sL[1])
        new_arr = Array(new[1], prime(sL[1]), sL[1])
        @test norm(old_arr - new_arr) < 1e-12
    end

    @testset "liouvillian_propagator_itensor forwards to liouvillian_propagator" begin
        old = @test_deprecated liouvillian_propagator_itensor(H, sL, dt)
        new = liouvillian_propagator(H, sL, dt)
        @test old isa ITensor
        @test new isa ITensor
        old_arr = Array(old, prime(sL[1]), sL[1])
        new_arr = Array(new, prime(sL[1]), sL[1])
        @test norm(old_arr - new_arr) < 1e-12
    end
end
