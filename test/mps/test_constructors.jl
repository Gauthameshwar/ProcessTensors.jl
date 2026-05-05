using Test
using ITensors
using ProcessTensors

@testset "MPS constructor forwarding API" begin
    @testset "random_mps forwarding (spin and boson)" begin
        spin_sites = siteinds("S=1/2", 4)
        boson_sites = siteinds("Boson", 3; dim=3)
        spin_state = fill("Up", length(spin_sites))
        boson_state = fill("0", length(boson_sites))

        # random_mps(sites; kwargs...)
        @test_nowarn random_mps(spin_sites; linkdims=2)
        @test_nowarn random_mps(boson_sites; linkdims=2)
        @test random_mps(spin_sites; linkdims=2) isa MPS{Hilbert}
        @test random_mps(boson_sites; linkdims=2) isa MPS{Hilbert}

        # random_mps(sites, state; kwargs...)
        @test_nowarn random_mps(spin_sites, spin_state; linkdims=2)
        @test_nowarn random_mps(boson_sites, boson_state; linkdims=2)
        @test random_mps(spin_sites, spin_state; linkdims=2) isa MPS{Hilbert}
        @test random_mps(boson_sites, boson_state; linkdims=2) isa MPS{Hilbert}

        # random_mps(eltype, sites; kwargs...)
        @test_nowarn random_mps(ComplexF64, spin_sites; linkdims=2)
        @test_nowarn random_mps(ComplexF64, boson_sites; linkdims=2)
        @test random_mps(ComplexF64, spin_sites; linkdims=2) isa MPS{Hilbert}
        @test random_mps(ComplexF64, boson_sites; linkdims=2) isa MPS{Hilbert}

        # random_mps(eltype, sites, state; kwargs...)
        @test_nowarn random_mps(ComplexF64, spin_sites, spin_state; linkdims=2)
        @test_nowarn random_mps(ComplexF64, boson_sites, boson_state; linkdims=2)
        @test random_mps(ComplexF64, spin_sites, spin_state; linkdims=2) isa MPS{Hilbert}
        @test random_mps(ComplexF64, boson_sites, boson_state; linkdims=2) isa MPS{Hilbert}
    end

    @testset "projector forwarding" begin
        spin_sites = siteinds("S=1/2", 4)
        boson_sites = siteinds("Boson", 3; dim=3)
        m_spin = MPS(spin_sites, fill("Up", length(spin_sites)))
        m_boson = MPS(boson_sites, fill("0", length(boson_sites)))

        @test_nowarn projector(m_spin)
        @test_nowarn projector(m_boson)
        @test projector(m_spin) isa MPO{Hilbert}
        @test projector(m_boson) isa MPO{Hilbert}
    end

    @testset "outer forwarding" begin
        spin_sites = siteinds("S=1/2", 4)
        boson_sites = siteinds("Boson", 3; dim=3)
        m1_spin = MPS(spin_sites, fill("Up", length(spin_sites)))
        m2_spin = MPS(spin_sites, fill("Dn", length(spin_sites)))
        m1_boson = MPS(boson_sites, fill("0", length(boson_sites)))
        m2_boson = MPS(boson_sites, fill("1", length(boson_sites)))

        # Prime one input to avoid ITensorMPS shared-index deprecation warning.
        @test_nowarn outer(m1_spin', m2_spin)
        @test_nowarn outer(m1_boson', m2_boson)
        @test outer(m1_spin', m2_spin) isa MPO{Hilbert}
        @test outer(m1_boson', m2_boson) isa MPO{Hilbert}
    end
end
