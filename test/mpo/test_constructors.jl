# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: test/mpo/test_constructors.jl
# Contributor: Gauthameshwar S.
#
# Tests that MPO constructor helpers forward to ITensorMPS and return wrapped
# Hilbert-space operators.
#
# Run with:
#   julia --project=. test/runtests.jl

using Test
using ITensors
using ProcessTensors

@testset "API surface: MPO constructor exports" begin
    @test :random_mpo ∈ names(ProcessTensors)
end

@testset "MPO constructor forwarding API" begin
    @testset "random_mpo forwarding (spin and boson)" begin
        spin_sites = siteinds("S=1/2", 4)
        boson_sites = siteinds("Boson", 3; dim=3)

        # random_mpo(sites; kwargs...)
        @test_nowarn random_mpo(spin_sites)
        @test_nowarn random_mpo(boson_sites)
        @test random_mpo(spin_sites) isa MPO{Hilbert}
        @test random_mpo(boson_sites) isa MPO{Hilbert}

    end
end
