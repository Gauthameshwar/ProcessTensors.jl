using Test
using ITensors
using ProcessTensors

@testset "Networks manipulations forwarding API" begin
    @testset "MPS manipulations (spin and boson)" begin
        spin_sites = siteinds("S=1/2", 4)
        boson_sites = siteinds("Boson", 3; dim=3)

        m_spin = MPS(spin_sites, fill("Up", length(spin_sites)))
        m_boson = MPS(boson_sites, fill("0", length(boson_sites)))

        phi_spin = m_spin[1] * m_spin[2]
        phi_boson = m_boson[1] * m_boson[2]

        @test_nowarn replacebond(m_spin, 1, phi_spin)
        @test_nowarn replacebond!(m_spin, 1, phi_spin)
        @test_nowarn swapbondsites(m_spin, 1)
        @test_nowarn movesite(m_spin, 1 => 3)
        @test_nowarn movesites(m_spin, [1 => 2, 2 => 3])

        @test_nowarn replacebond(m_boson, 1, phi_boson)
        @test_nowarn replacebond!(m_boson, 1, phi_boson)
        @test_nowarn swapbondsites(m_boson, 1)
        @test_nowarn movesite(m_boson, 1 => 2)
        @test_nowarn movesites(m_boson, [1 => 2])
    end
end
