using Test
using ITensors
using ProcessTensors

@testset "Networks indices forwarding API" begin
    @testset "Query functions (single-object and pairwise)" begin
        spin_sites = siteinds("S=1/2", 4)
        boson_sites = siteinds("Boson", 3; dim=3)

        m_spin = MPS(spin_sites, fill("Up", length(spin_sites)))
        m_spin_2 = copy(m_spin)
        o_spin = MPO(spin_sites, "Id")
        m_boson = MPS(boson_sites, fill("0", length(boson_sites)))
        m_boson_2 = copy(m_boson)
        o_boson = MPO(boson_sites, "Id")

        @test_nowarn siteinds(m_spin)
        @test_nowarn siteind(m_spin, 1)
        @test_nowarn linkinds(m_spin)
        @test_nowarn linkind(m_spin, 1)
        @test_nowarn linkdim(m_spin, 1)
        @test_nowarn linkdims(m_spin)
        @test_nowarn maxlinkdim(m_spin)
        @test_nowarn totalqn(m_spin)
        @test_nowarn findfirstsiteind(m_spin, siteind(m_spin, 1))
        @test_nowarn findfirstsiteinds(m_spin, siteind(m_spin, 1))
        @test_nowarn findsite(m_spin, siteind(m_spin, 1))
        @test_nowarn findsite(m_spin, 1)
        @test_nowarn findsites(m_spin, siteind(m_spin, 1))
        @test_nowarn findsites(m_spin, 1)
        @test_nowarn firstsiteind(m_spin, 1)
        @test_nowarn firstsiteinds(o_spin)
        @test_nowarn common_siteind(m_spin, m_spin_2, 1)
        @test_nowarn common_siteind(m_spin, m_spin_2.core, 1)
        @test_nowarn common_siteinds(m_spin, m_spin_2)
        @test_nowarn unique_siteind(m_spin, m_spin_2, 1)
        @test_nowarn unique_siteinds(m_spin, m_spin_2)
        @test_nowarn hassameinds(m_spin, m_spin_2)
        @test_nowarn hassameinds(m_spin, m_spin_2.core)

        @test_nowarn siteinds(m_boson)
        @test_nowarn siteind(m_boson, 1)
        @test_nowarn linkinds(m_boson)
        @test_nowarn linkind(m_boson, 1)
        @test_nowarn linkdim(m_boson, 1)
        @test_nowarn linkdims(m_boson)
        @test_nowarn maxlinkdim(m_boson)
        @test_nowarn totalqn(m_boson)
        @test_nowarn findfirstsiteind(m_boson, siteind(m_boson, 1))
        @test_nowarn findfirstsiteinds(m_boson, siteind(m_boson, 1))
        @test_nowarn findsite(m_boson, siteind(m_boson, 1))
        @test_nowarn findsite(m_boson, 1)
        @test_nowarn findsites(m_boson, siteind(m_boson, 1))
        @test_nowarn findsites(m_boson, 1)
        @test_nowarn firstsiteind(m_boson, 1)
        @test_nowarn firstsiteinds(o_boson)
        @test_nowarn common_siteind(m_boson, m_boson_2, 1)
        @test_nowarn common_siteind(m_boson, m_boson_2.core, 1)
        @test_nowarn common_siteinds(m_boson, m_boson_2)
        @test_nowarn unique_siteind(m_boson, m_boson_2, 1)
        @test_nowarn unique_siteinds(m_boson, m_boson_2)
        @test_nowarn hassameinds(m_boson, m_boson_2)
        @test_nowarn hassameinds(m_boson, m_boson_2.core)
    end

    @testset "Out-of-place index transforms" begin
        spin_sites = siteinds("S=1/2", 4)
        boson_sites = siteinds("Boson", 3; dim=3)

        m_spin = MPS(spin_sites, fill("Up", length(spin_sites)))
        m_boson = MPS(boson_sites, fill("0", length(boson_sites)))

        spin_new_sites = prime.(siteinds(m_spin))
        boson_new_sites = prime.(siteinds(m_boson))

        @test_nowarn replace_siteinds(m_spin, spin_new_sites)
        @test_nowarn replaceprime(m_spin, 0 => 1)
        @test replace_siteinds(m_spin, spin_new_sites) isa MPS{Hilbert}
        @test replaceprime(m_spin, 0 => 1) isa MPS{Hilbert}

        @test_nowarn replace_siteinds(m_boson, boson_new_sites)
        @test_nowarn replaceprime(m_boson, 0 => 1)
        @test replace_siteinds(m_boson, boson_new_sites) isa MPS{Hilbert}
        @test replaceprime(m_boson, 0 => 1) isa MPS{Hilbert}
    end

    @testset "In-place index transforms" begin
        spin_sites = siteinds("S=1/2", 4)
        boson_sites = siteinds("Boson", 3; dim=3)

        m_spin = MPS(spin_sites, fill("Up", length(spin_sites)))
        m_boson = MPS(boson_sites, fill("0", length(boson_sites)))

        spin_new_sites = prime.(siteinds(m_spin))
        boson_new_sites = prime.(siteinds(m_boson))

        @test_nowarn replace_siteinds!(m_spin, spin_new_sites)
        @test_nowarn replace_siteinds!(m_boson, boson_new_sites)
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
