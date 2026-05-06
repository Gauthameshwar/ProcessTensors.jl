using Test
using ITensors
using ProcessTensors

@testset "Networks algebra forwarding API" begin
    @testset "MPS algebra methods (spin and boson)" begin
        spin_sites = siteinds("S=1/2", 4)
        boson_sites = siteinds("Boson", 3; dim=3)

        m_spin = MPS(spin_sites, fill("Up", length(spin_sites)))
        m_spin_2 = MPS(spin_sites, fill("Dn", length(spin_sites)))
        op_spin = projector(m_spin)

        m_boson = MPS(boson_sites, fill("0", length(boson_sites)))
        m_boson_2 = MPS(boson_sites, fill("1", length(boson_sites)))
        op_boson = projector(m_boson)

        @test_nowarn error_contract(m_spin, op_spin, m_spin_2)
        @test_nowarn error_contract(m_spin, m_spin_2, op_spin)
        @test_nowarn truncate(m_spin; maxdim=2)
        @test_nowarn truncate!(m_spin; maxdim=2)

        @test_nowarn apply(op_spin, m_spin)
        @test_nowarn contract(op_spin, m_spin)
        @test_nowarn add(m_spin, m_spin_2)
        @test_nowarn add(m_spin.core, m_spin)
        @test_nowarn m_spin + m_spin_2
        @test_nowarn m_spin - m_spin_2
        @test_nowarn 2.0 * m_spin
        @test_nowarn m_spin * 2.0

        @test_nowarn error_contract(m_boson, op_boson, m_boson_2)
        @test_nowarn error_contract(m_boson, m_boson_2, op_boson)
        @test_nowarn truncate(m_boson; maxdim=2)
        @test_nowarn truncate!(m_boson; maxdim=2)

        @test_nowarn apply(op_boson, m_boson)
        @test_nowarn contract(op_boson, m_boson)
        @test_nowarn add(m_boson, m_boson_2)
        @test_nowarn add(m_boson.core, m_boson)
        @test_nowarn m_boson + m_boson_2
        @test_nowarn m_boson - m_boson_2
        @test_nowarn 2.0 * m_boson
        @test_nowarn m_boson * 2.0
    end
end
