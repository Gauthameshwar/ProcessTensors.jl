using Test
using ITensors
using ProcessTensors

@testset "MPS observable forwarding API" begin
    @testset "Two-state scalar observables" begin
        spin_sites = siteinds("S=1/2", 4)
        boson_sites = siteinds("Boson", 3; dim=3)

        m_spin_1 = MPS(spin_sites, fill("Up", length(spin_sites)))
        m_spin_2 = MPS(spin_sites, fill("Dn", length(spin_sites)))
        m_boson_1 = MPS(boson_sites, fill("0", length(boson_sites)))
        m_boson_2 = MPS(boson_sites, fill("1", length(boson_sites)))

        @test_nowarn inner(m_spin_1, m_spin_2)
        @test_nowarn dot(m_spin_1, m_spin_2)
        @test_nowarn loginner(m_spin_1, m_spin_2)
        @test_nowarn logdot(m_spin_1, m_spin_2)

        @test_nowarn inner(m_boson_1, m_boson_2)
        @test_nowarn dot(m_boson_1, m_boson_2)
        @test_nowarn loginner(m_boson_1, m_boson_2)
        @test_nowarn logdot(m_boson_1, m_boson_2)
    end

    @testset "Three-argument inner forwarding" begin
        spin_sites = siteinds("S=1/2", 4)
        boson_sites = siteinds("Boson", 3; dim=3)

        m_spin = MPS(spin_sites, fill("Up", length(spin_sites)))
        m_boson = MPS(boson_sites, fill("0", length(boson_sites)))

        a_spin = projector(m_spin)
        a_boson = projector(m_boson)

        # Prime bra MPS to match ITensorMPS non-deprecated index convention.
        @test_nowarn inner(m_spin', a_spin, m_spin)
        @test_nowarn inner(m_boson', a_boson, m_boson)
    end

    @testset "Single-state observables" begin
        spin_sites = siteinds("S=1/2", 4)
        boson_sites = siteinds("Boson", 3; dim=3)

        m_spin = MPS(spin_sites, fill("Up", length(spin_sites)))
        m_boson = MPS(boson_sites, fill("0", length(boson_sites)))

        @test_nowarn norm(m_spin)
        @test_nowarn lognorm(m_spin)
        @test_nowarn expect(m_spin, "Sz")
        @test_nowarn correlation_matrix(m_spin, "Sz", "Sz")
        @test_nowarn entropy(m_spin, 2)

        @test_nowarn norm(m_boson)
        @test_nowarn lognorm(m_boson)
        @test_nowarn expect(m_boson, "N")
        @test_nowarn correlation_matrix(m_boson, "N", "N")
        @test_nowarn entropy(m_boson, 2)
    end

    @testset "Sampling forwarding" begin
        spin_sites = siteinds("S=1/2", 4)
        boson_sites = siteinds("Boson", 3; dim=3)

        m_spin = MPS(spin_sites, fill("Up", length(spin_sites)))
        m_boson = MPS(boson_sites, fill("0", length(boson_sites)))

        @test_nowarn orthogonalize!(m_spin, 1)
        @test_nowarn sample(m_spin)
        @test_nowarn sample!(m_spin)

        @test_nowarn orthogonalize!(m_boson, 1)
        @test_nowarn sample(m_boson)
        @test_nowarn sample!(m_boson)
    end
end
