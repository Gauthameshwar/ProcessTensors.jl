# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: test/networks/test_indices.jl
# Contributor: Gauthameshwar S.
#
# Tests that compact network index helpers forward correctly on wrapped MPS and
# MPO objects.
#
# Run with:
#   julia --project=. test/runtests.jl

using Test
using ITensors
using ProcessTensors

@testset "API surface: exported index helpers" begin
    for name in (
        :siteinds, :siteind, :linkinds, :linkind, :linkdim, :linkdims, :maxlinkdim,
        :OpSum, :op,
    )
        @test name ∈ names(ProcessTensors)
    end
    @test ProcessTensors.siteinds === ProcessTensors.ITensorMPS.siteinds
end

@testset "indices forwarding API" begin
    @testset "Query functions" begin
        spin_sites = siteinds("S=1/2", 4)
        boson_sites = siteinds("Boson", 3; dim=3)

        m_spin = MPS(spin_sites, fill("Up", length(spin_sites)))
        o_spin = MPO(spin_sites, "Id")
        m_boson = MPS(boson_sites, fill("0", length(boson_sites)))
        o_boson = MPO(boson_sites, "Id")

        @test_nowarn siteinds(m_spin)
        @test_nowarn siteind(m_spin, 1)
        @test_nowarn linkinds(m_spin)
        @test_nowarn linkind(m_spin, 1)
        @test_nowarn linkdim(m_spin, 1)
        @test_nowarn linkdims(m_spin)
        @test_nowarn maxlinkdim(m_spin)
        @test_nowarn siteinds(o_spin)

        @test_nowarn siteinds(m_boson)
        @test_nowarn siteind(m_boson, 1)
        @test_nowarn linkinds(m_boson)
        @test_nowarn linkind(m_boson, 1)
        @test_nowarn linkdim(m_boson, 1)
        @test_nowarn linkdims(m_boson)
        @test_nowarn maxlinkdim(m_boson)
        @test_nowarn siteinds(o_boson)
    end

    @testset "Index tag helper queries" begin
        spin_sites = siteinds("S=1/2", 4)
        boson_sites = siteinds("Boson", 3; dim=3)

        s_spin = spin_sites[1]
        s_boson = boson_sites[1]

        @test_nowarn tag_tokens(s_spin)
        @test_nowarn has_tag_token(s_spin, "Site")
        @test_nowarn has_tag_prefix(s_spin, "S=")
        @test tag_value(s_spin, "n=") == "1"
        @test tag_value(s_spin, "missing_prefix=") === nothing

        @test_nowarn tag_tokens(s_boson)
        @test_nowarn has_tag_token(s_boson, "Site")
        @test_nowarn has_tag_prefix(s_boson, "n=")
        @test_nowarn tag_value(s_boson, "n=")
    end
end
