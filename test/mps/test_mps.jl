using Test
using ITensors
using ProcessTensors

@testset "MPS forwarding API" begin
    @testset "Structural definitions (Hilbert and Liouville)" begin
        spin_sites = siteinds("S=1/2", 4)
        core_spin = ProcessTensors.ITensorMPS.MPS(spin_sites, fill("Up", 4))

        m_h = MPS{Hilbert}(core_spin)
        @test m_h isa MPS{Hilbert}
        @test m_h.core isa ProcessTensors.ITensorMPS.MPS
        @test m_h.combiners === nothing

        combiners = [ITensor(1.0) for _ in 1:length(spin_sites)]
        m_l = MPS{Liouville}(copy(core_spin), combiners)
        @test m_l isa MPS{Liouville}
        @test m_l.core isa ProcessTensors.ITensorMPS.MPS
        @test m_l.combiners === combiners
    end

    @testset "Outer constructors (spin and boson examples)" begin
        spin_sites = siteinds("S=1/2", 4)
        boson_sites = siteinds("Boson", 3; dim=4)
        spin_states = fill("Up", length(spin_sites))
        boson_states = fill("0", length(boson_sites))

        # MPS(args...; kwargs...)
        @test_nowarn MPS(spin_sites, spin_states)
        @test_nowarn MPS(boson_sites, boson_states)
        @test MPS(spin_sites, spin_states) isa MPS{Hilbert}
        @test MPS(boson_sites, boson_states) isa MPS{Hilbert}

        # MPS{Hilbert}(args...; kwargs...)
        @test_nowarn MPS{Hilbert}(spin_sites, spin_states)
        @test_nowarn MPS{Hilbert}(boson_sites, boson_states)
        @test MPS{Hilbert}(spin_sites, spin_states) isa MPS{Hilbert}
        @test MPS{Hilbert}(boson_sites, boson_states) isa MPS{Hilbert}

        # Prepare coefficient arrays for AbstractArray constructors
        spin_coeffs = randn(ComplexF64, 2^length(spin_sites))
        boson_coeffs = randn(ComplexF64, 4^length(boson_sites))

        # MPS(A::AbstractArray, ...)
        @test_nowarn MPS(spin_coeffs, spin_sites)
        @test_nowarn MPS(boson_coeffs, boson_sites)
        @test MPS(spin_coeffs, spin_sites) isa MPS{Hilbert}
        @test MPS(boson_coeffs, boson_sites) isa MPS{Hilbert}

        # MPS{Hilbert}(A::AbstractArray, ...)
        @test_nowarn MPS{Hilbert}(spin_coeffs, spin_sites)
        @test_nowarn MPS{Hilbert}(boson_coeffs, boson_sites)
        @test MPS{Hilbert}(spin_coeffs, spin_sites) isa MPS{Hilbert}
        @test MPS{Hilbert}(boson_coeffs, boson_sites) isa MPS{Hilbert}

        # MPS(A::ITensor, sites; kwargs...)
        psi_spin = randomITensor(spin_sites...)
        psi_boson = randomITensor(boson_sites...)
        @test_nowarn MPS(psi_spin, spin_sites)
        @test_nowarn MPS(psi_boson, boson_sites)
        @test MPS(psi_spin, spin_sites) isa MPS{Hilbert}
        @test MPS(psi_boson, boson_sites) isa MPS{Hilbert}

        # MPS{Hilbert}(A::ITensor, sites; kwargs...)
        @test_nowarn MPS{Hilbert}(psi_spin, spin_sites)
        @test_nowarn MPS{Hilbert}(psi_boson, boson_sites)
        @test MPS{Hilbert}(psi_spin, spin_sites) isa MPS{Hilbert}
        @test MPS{Hilbert}(psi_boson, boson_sites) isa MPS{Hilbert}

        # MPS{Liouville}(combiners, args...)
        spin_combiners = [ITensor(1.0) for _ in 1:length(spin_sites)]
        boson_combiners = [ITensor(1.0) for _ in 1:length(boson_sites)]

        @test_nowarn MPS{Liouville}(spin_combiners, spin_sites, spin_states)
        @test_nowarn MPS{Liouville}(boson_combiners, boson_sites, boson_states)
        @test MPS{Liouville}(spin_combiners, spin_sites, spin_states) isa MPS{Liouville}
        @test MPS{Liouville}(boson_combiners, boson_sites, boson_states) isa MPS{Liouville}

        # MPS{Liouville}(combiners, A::AbstractArray, args...)
        @test_nowarn MPS{Liouville}(spin_combiners, spin_coeffs, spin_sites)
        @test_nowarn MPS{Liouville}(boson_combiners, boson_coeffs, boson_sites)
        @test MPS{Liouville}(spin_combiners, spin_coeffs, spin_sites) isa MPS{Liouville}
        @test MPS{Liouville}(boson_combiners, boson_coeffs, boson_sites) isa MPS{Liouville}
    end

    @testset "Delegated getproperty and setproperty" begin
        spin_sites = siteinds("S=1/2", 4)
        m = MPS(spin_sites, fill("Up", 4))
        core = m.core

        common_props = filter(sym -> hasproperty(core, sym), (:llim, :rlim, :center, :orthocenter))
        @test !isempty(common_props)

        for prop in common_props
            @test_nowarn getproperty(m, prop)
        end

        # Set to current values to check forwarding into ITensorMPS without mutation side effects.
        for prop in common_props
            val = getproperty(m, prop)
            @test_nowarn setproperty!(m, prop, val)
            @test getproperty(m, prop) == val
        end
    end

    @testset "Forwarded indexing and show" begin
        spin_sites = siteinds("S=1/2", 4)
        m = MPS(spin_sites, fill("Up", 4))

        @test length(m) == 4
        @test_nowarn m[1]
        @test_nowarn m[1] = m[1]

        out = sprint(show, m)
        @test occursin("ProcessTensors.MPS", out)
        @test occursin("Space: Hilbert", out)
        @test occursin("Sites: 4", out)
    end
end
