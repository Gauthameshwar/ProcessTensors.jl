using Test
using ITensors
using ProcessTensors

@testset "Networks orthogonality forwarding API" begin
    @testset "MPS orthogonality methods (spin and boson)" begin
        spin_sites = siteinds("S=1/2", 4)
        boson_sites = siteinds("Boson", 3; dim=3)

        m_spin = MPS(spin_sites, fill("Up", length(spin_sites)))
        m_boson = MPS(boson_sites, fill("0", length(boson_sites)))

        @test_nowarn orthogonalize!(m_spin, 2)
        @test_nowarn isortho(m_spin)
        @test_nowarn ortho_lims(m_spin)
        @test_nowarn orthocenter(m_spin)
        @test_nowarn orthogonalize(m_spin, 2)
        @test_nowarn normalize!(m_spin)
        @test_nowarn set_ortho_lims!(m_spin, 1:length(m_spin))
        @test_nowarn reset_ortho_lims!(m_spin)

        @test_nowarn orthogonalize!(m_boson, 2)
        @test_nowarn isortho(m_boson)
        @test_nowarn ortho_lims(m_boson)
        @test_nowarn orthocenter(m_boson)
        @test_nowarn orthogonalize(m_boson, 2)
        @test_nowarn normalize!(m_boson)
        @test_nowarn set_ortho_lims!(m_boson, 1:length(m_boson))
        @test_nowarn reset_ortho_lims!(m_boson)
    end
end
