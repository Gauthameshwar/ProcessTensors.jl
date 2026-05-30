using Test
using ITensors
using ProcessTensors

@testset "MPO forwarding API" begin
    @testset "Structural definitions (Hilbert and Liouville)" begin
        spin_sites = siteinds("S=1/2", 4)
        core_spin = ProcessTensors.ITensorMPS.MPO(spin_sites, "Id")

        m_h = MPO{Hilbert}(core_spin)
        @test m_h isa MPO{Hilbert}
        @test m_h.core isa ProcessTensors.ITensorMPS.MPO
        @test m_h.combiners === nothing

        combiners = [ITensor(1.0) for _ in 1:length(spin_sites)]
        m_l = MPO{Liouville}(copy(core_spin), combiners)
        @test m_l isa MPO{Liouville}
        @test m_l.core isa ProcessTensors.ITensorMPS.MPO
        @test m_l.combiners === combiners
    end

    @testset "Outer constructors (spin and boson examples)" begin
        spin_sites = siteinds("S=1/2", 4)
        boson_sites = siteinds("Boson", 3; dim=3)

        # MPO(args...; kwargs...)
        @test_nowarn MPO(spin_sites, "Id")
        @test_nowarn MPO(boson_sites, "Id")
        @test MPO(spin_sites, "Id") isa MPO{Hilbert}
        @test MPO(boson_sites, "Id") isa MPO{Hilbert}

        # MPO{Hilbert}(args...; kwargs...) — explicit typed vararg outer constructors
        m_h_varargs = MPO{Hilbert}(spin_sites, "Id")
        @test_nowarn m_h_varargs
        @test m_h_varargs isa MPO{Hilbert}
        @test_nowarn MPO{Hilbert}(boson_sites, "Id")
        @test MPO{Hilbert}(spin_sites, "Id") isa MPO{Hilbert}
        @test MPO{Hilbert}(boson_sites, "Id") isa MPO{Hilbert}

        # For MPO(AbstractArray, ...), pass paired output/input site index sets.
        spin_site_pairs = IndexSet.(prime.(spin_sites), dag.(spin_sites))
        boson_site_pairs = IndexSet.(prime.(boson_sites), dag.(boson_sites))
        spin_coeffs = randn(ComplexF64, prod(dim, spin_site_pairs))
        boson_coeffs = randn(ComplexF64, prod(dim, boson_site_pairs))

        # MPO(A::AbstractArray, ...)
        @test_nowarn MPO(spin_coeffs, spin_site_pairs)
        @test_nowarn MPO(boson_coeffs, boson_site_pairs)
        @test MPO(spin_coeffs, spin_site_pairs) isa MPO{Hilbert}
        @test MPO(boson_coeffs, boson_site_pairs) isa MPO{Hilbert}

        # MPO{Hilbert}(A::AbstractArray, ...)
        @test_nowarn MPO{Hilbert}(spin_coeffs, spin_site_pairs)
        @test_nowarn MPO{Hilbert}(boson_coeffs, boson_site_pairs)
        @test MPO{Hilbert}(spin_coeffs, spin_site_pairs) isa MPO{Hilbert}
        @test MPO{Hilbert}(boson_coeffs, boson_site_pairs) isa MPO{Hilbert}

        # MPO(A::ITensor, sites; kwargs...)
        op_spin = random_itensor(prime.(spin_sites)..., dag.(spin_sites)...)
        op_boson = random_itensor(prime.(boson_sites)..., dag.(boson_sites)...)
        @test_nowarn MPO(op_spin, spin_sites)
        @test_nowarn MPO(op_boson, boson_sites)
        @test MPO(op_spin, spin_sites) isa MPO{Hilbert}
        @test MPO(op_boson, boson_sites) isa MPO{Hilbert}

        # MPO{Hilbert}(A::ITensor, sites; kwargs...)
        @test_nowarn MPO{Hilbert}(op_spin, spin_sites)
        @test_nowarn MPO{Hilbert}(op_boson, boson_sites)
        @test MPO{Hilbert}(op_spin, spin_sites) isa MPO{Hilbert}
        @test MPO{Hilbert}(op_boson, boson_sites) isa MPO{Hilbert}

        # MPO{Liouville}(combiners, args...)
        spin_combiners = [ITensor(1.0) for _ in 1:length(spin_sites)]
        boson_combiners = [ITensor(1.0) for _ in 1:length(boson_sites)]

        m_l_varargs = MPO{Liouville}(spin_combiners, spin_sites, "Id")
        @test_nowarn m_l_varargs
        @test m_l_varargs isa MPO{Liouville}
        @test_nowarn MPO{Liouville}(boson_combiners, boson_sites, "Id")
        @test MPO{Liouville}(spin_combiners, spin_sites, "Id") isa MPO{Liouville}
        @test MPO{Liouville}(boson_combiners, boson_sites, "Id") isa MPO{Liouville}

        # MPO{Liouville}(combiners, A::AbstractArray, args...)
        @test_nowarn MPO{Liouville}(spin_combiners, spin_coeffs, spin_site_pairs)
        @test_nowarn MPO{Liouville}(boson_combiners, boson_coeffs, boson_site_pairs)
        @test MPO{Liouville}(spin_combiners, spin_coeffs, spin_site_pairs) isa MPO{Liouville}
        @test MPO{Liouville}(boson_combiners, boson_coeffs, boson_site_pairs) isa MPO{Liouville}
    end

    @testset "Delegated getproperty and setproperty" begin
        spin_sites = siteinds("S=1/2", 4)
        m = MPO(spin_sites, "Id")
        core = m.core

        common_props = filter(sym -> hasproperty(core, sym), (:llim, :rlim, :center, :orthocenter))
        @test !isempty(common_props)

        for prop in common_props
            @test_nowarn getproperty(m, prop)
        end

        for prop in common_props
            val = getproperty(m, prop)
            @test_nowarn setproperty!(m, prop, val)
            @test getproperty(m, prop) == val
        end
    end

    @testset "Forwarded indexing, copy, and show" begin
        spin_sites = siteinds("S=1/2", 4)
        m_h = MPO(spin_sites, "Id")
        combiners = [ITensor(1.0) for _ in 1:length(spin_sites)]
        m_l = MPO{Liouville}(combiners, spin_sites, "Id")

        @test length(m_h) == 4
        @test_nowarn m_h[1]
        @test_nowarn m_h[1] = m_h[1]

        @test_nowarn copy(m_h)
        @test_nowarn copy(m_l)
        @test copy(m_h) isa MPO{Hilbert}
        @test copy(m_l) isa MPO{Liouville}

        out_h = sprint(show, m_h)
        out_l = sprint(show, m_l)
        plain_h = sprint(show, MIME"text/plain"(), m_h)
        plain_l = sprint(show, MIME"text/plain"(), m_l)
        @test out_h == plain_h
        @test out_l == plain_l
        for (out, space, combiner_line) in (
            (out_h, "Hilbert", "combiners: none"),
            (out_l, "Liouville", "4 ITensors"),
        )
            @test occursin("4-element MPO{$space}", out)
            @test occursin("site dims:", out)
            @test occursin("link dims:", out)
            @test occursin("maxlinkdim:", out)
            @test occursin(combiner_line, out)
            @test occursin("tensors:", out)
            @test occursin("[1]", out)
            @test occursin("[4]", out)
        end

        spin_sites_big = siteinds("S=1/2", 8)
        m_h_big = MPO(spin_sites_big, "Id")
        combiners_big = [ITensor(1.0) for _ in 1:length(spin_sites_big)]
        m_l_big = MPO{Liouville}(combiners_big, spin_sites_big, "Id")

        for (out, space, combiner_line) in (
            (sprint(show, m_h_big), "Hilbert", "combiners: none"),
            (sprint(show, m_l_big), "Liouville", "8 ITensors"),
        )
            @test occursin("8-element MPO{$space}", out)
            @test occursin("⋮", out)
            @test occursin("[1]", out)
            @test occursin("[2]", out)
            @test occursin("[7]", out)
            @test occursin("[8]", out)
            @test occursin(combiner_line, out)
        end
    end
end
