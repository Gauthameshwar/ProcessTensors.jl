# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: test/mpo/test_observables.jl
# Contributor: Gauthameshwar S.
#
# Tests that MPO observable helpers forward correctly on wrapped Hilbert-space
# operators.
#
# Run with:
#   julia --project=. test/runtests.jl

using Test
using ITensors
using ProcessTensors

@testset "API surface: MPO observable exports" begin
    @test :tr ∈ names(ProcessTensors)
end

@testset "MPO observable forwarding API" begin
    @testset "trace forwarding (spin and boson)" begin
        spin_sites = siteinds("S=1/2", 4)
        boson_sites = siteinds("Boson", 3; dim=3)

        mpo_spin = MPO(spin_sites, "Id")
        mpo_boson = MPO(boson_sites, "Id")

        @test_nowarn tr(mpo_spin)
        @test_nowarn tr(mpo_boson)
    end
end
