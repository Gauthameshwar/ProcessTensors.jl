using ProcessTensors
using ITensors
using Test

# Recover the built-in site family tag, e.g. `Site,S=1/2,n=1` -> `S=1/2`.
function physical_site_type_tag(s::Index)
    for token in tag_tokens(s)
        if token == "Site" || startswith(token, "n=") || startswith(token, "l=")
            continue
        end
        return token
    end
    return nothing
end

@testset "liouvillian.jl: liouv_sites site-type coverage" begin
    site_builders = [
        () -> siteinds("S=1/2", 2),
        () -> siteinds("Fermion", 2),
        () -> siteinds("Boson", 2; dim=4),
        () -> siteinds("Electron", 2),
        () -> siteinds("Qudit", 2; dim=3),
    ]

    for make_sites in site_builders
        sites = make_sites()
        sL = liouv_sites(sites)

        @test length(sL) == length(sites)
        for i in eachindex(sites)
            site_type = physical_site_type_tag(sites[i])
            @test dim(sL[i]) == dim(sites[i])^2
            @test hastags(sL[i], "Liouv")
            @test site_type !== nothing
            @test hastags(sL[i], "ptype=$site_type")
        end
    end
end

@testset "liouvillian.jl: legacy phys tags still read" begin
    legacy_liouv_site = Index(4; tags="Liouv,phys=S=1/2,n=1")
    Sz_left = @test_nowarn op("Sz_L", legacy_liouv_site)

    @test hasind(Sz_left, legacy_liouv_site)
    @test hasind(Sz_left, prime(legacy_liouv_site))
end

@testset "liouvillian.jl: dm/liouville/hilbert conversions" begin
    sites = siteinds("S=1/2", 2)
    sL = liouv_sites(sites)

    ψ1 = MPS(sites, ["Up", "Dn"])
    ψ2 = MPS(sites, ["Dn", "Up"])

    ρ_pure = to_dm(ψ1)
    @test ρ_pure isa MPO{Hilbert}

    ρ_pure_liou = to_liouville(ρ_pure; sites=sL)
    @test ρ_pure_liou isa MPS{Liouville}
    @test length(ρ_pure_liou.combiners) == length(sites)

    ρ_pure_back = to_hilbert(ρ_pure_liou)
    @test ρ_pure_back isa MPO{Hilbert}
    @test length(siteinds(ρ_pure_back)) == length(sites)

    ρ_mixed = to_dm([ψ1, ψ2]; coeffs=[0.3, 0.7])
    @test ρ_mixed isa MPO{Hilbert}

    ρ_mixed_liou = to_liouville(ρ_mixed; sites=sL)
    @test ρ_mixed_liou isa MPS{Liouville}
    @test length(ρ_mixed_liou.combiners) == length(sites)

    ρ_mixed_back = to_hilbert(ρ_mixed_liou)
    @test ρ_mixed_back isa MPO{Hilbert}
    @test length(siteinds(ρ_mixed_back)) == length(sites)
end
