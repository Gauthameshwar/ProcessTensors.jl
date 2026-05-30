using ProcessTensors
using ProcessTensors.Spectrals: ohmic_sd
using ITensors
using Test

@testset "environments.jl: mode type and field definitions" begin
    @test BosonicMode <: AbstractBathMode
    @test SpinMode <: AbstractBathMode
    @test BosonicBath <: AbstractBath
    @test SpinBath <: AbstractBath

    @test hasfield(BosonicMode, :rho0)
    @test hasfield(BosonicMode, :H)
    @test hasfield(BosonicMode, :n_max)
    @test hasfield(BosonicMode, :sites)
    @test fieldtype(BosonicMode, :H) == OpSum
    @test fieldtype(BosonicMode, :n_max) == Int
    @test fieldtype(BosonicMode, :sites) == Vector{Index}

    @test hasfield(SpinMode, :rho0)
    @test hasfield(BosonicMode, :coupling)
    @test hasfield(SpinMode, :H)
    @test hasfield(SpinMode, :coupling)
    @test hasfield(SpinMode, :sites)
    @test fieldtype(SpinMode, :H) == OpSum
    @test fieldtype(SpinMode, :sites) == Vector{Index}
end

@testset "environments.jl: bosonic and spin mode constructors" begin
    b_sites = liouv_sites(siteinds("Boson", 1; dim=4))
    s_sites = liouv_sites(siteinds("S=1/2", 1))
    rho_b = random_mps(b_sites)
    rho_s = random_mps(s_sites)
    n_max_b = dim(only(b_sites)) - 1

    H_b = OpSum() + (0.3, "N", 1)
    H_s = OpSum() + (0.5, "Sz", 1)

    @test_nowarn BosonicMode(b_sites, H_b, n_max_b, rho_b)
    @test_nowarn BosonicMode(b_sites, H_b, rho_b; n_max=n_max_b)
    @test_nowarn BosonicMode(sites=b_sites, H=H_b, rho0=rho_b, n_max=n_max_b)
    @test_nowarn SpinMode(s_sites, H_s, rho_s)
    @test_nowarn SpinMode(sites=s_sites, H=H_s, rho0=rho_s)

    bm = bosonic_mode(b_sites, H_b, n_max_b, rho_b)
    sm = spin_mode(s_sites, H_s, rho_s)
    @test bm isa BosonicMode
    @test sm isa SpinMode
    @test bm.n_max == n_max_b
    @test length(bm.sites) == 1
    @test length(sm.sites) == 1
end

@testset "environments.jl: mode constructor validation errors/warnings" begin
    b_sites = liouv_sites(siteinds("Boson", 1; dim=4))
    s_sites = liouv_sites(siteinds("S=1/2", 1))
    rho_b = random_mps(b_sites)
    rho_s = random_mps(s_sites)
    n_max_b = dim(only(b_sites)) - 1

    bad_b_sites = liouv_sites(siteinds("Boson", 2; dim=4))
    bad_s_sites = liouv_sites(siteinds("S=1/2", 2))

    @test_throws ArgumentError BosonicMode(bad_b_sites, OpSum(), 3, rho_b)
    @test_throws ArgumentError SpinMode(bad_s_sites, OpSum(), rho_s)
    @test_throws ArgumentError BosonicMode(b_sites, OpSum(), 2, rho_b)
    @test_throws ArgumentError SpinMode(s_sites, OpSum(), rho_b)

    @test_warn r"BosonicMode:H is empty" BosonicMode(b_sites, OpSum(), n_max_b, rho_b)
    @test_warn r"SpinMode:H is empty" SpinMode(s_sites, OpSum(), rho_s)
end

@testset "environments.jl: bath constructors and aliases" begin
    sd = ohmic_sd()

    b1_sites = liouv_sites(siteinds("Boson", 1; dim=4))
    b2_sites = liouv_sites(siteinds("Boson", 1; dim=5))
    s1_sites = liouv_sites(siteinds("S=1/2", 1))
    s2_sites = liouv_sites(siteinds("S=1/2", 1))

    b1 = bosonic_mode(b1_sites, OpSum() + (0.1, "N", 1), dim(only(b1_sites)) - 1, random_mps(b1_sites); coupling=OpSum() + (0.2, "N", 1, "Sz", 2))
    b2 = bosonic_mode(b2_sites, OpSum() + (0.1, "N", 1), dim(only(b2_sites)) - 1, random_mps(b2_sites); coupling=OpSum() + (0.2, "N", 1, "Sz", 2))
    s1 = spin_mode(s1_sites, OpSum() + (0.1, "Sz", 1), random_mps(s1_sites); coupling=OpSum() + (0.2, "Sz", 1, "Sz", 2))
    s2 = spin_mode(s2_sites, OpSum() + (0.1, "Sz", 1), random_mps(s2_sites); coupling=OpSum() + (0.2, "Sz", 1, "Sz", 2))

    @test_nowarn BosonicBath([b1, b2], sd)
    @test_nowarn SpinBath([s1, s2], sd)
    @test_nowarn bosonic_bath([b1, b2]; spectral_density=sd)
    @test_nowarn spin_bath([s1, s2]; spectral_density=sd)

    bb = BosonicBath([b1, b2], sd)
    sb = SpinBath([s1, s2], sd)
    @test bb isa BosonicBath
    @test sb isa SpinBath
    @test length(bb.modes) == 2
    @test length(sb.modes) == 2
    @test length(mode_initial_states(bb)) == 2
    @test length(mode_initial_states(sb)) == 2
end

@testset "environments.jl: bath validation errors/warnings" begin
    sd = ohmic_sd()
    b_sites = liouv_sites(siteinds("Boson", 1; dim=4))
    s_sites = liouv_sites(siteinds("S=1/2", 1))
    bm = bosonic_mode(b_sites, OpSum() + (0.1, "N", 1), dim(only(b_sites)) - 1, random_mps(b_sites))
    sm = spin_mode(s_sites, OpSum() + (0.1, "Sz", 1), random_mps(s_sites))

    @test_throws ArgumentError BosonicBath([only(b_sites), Index(2)], [bm], sd, OpSum())
    @test_throws ArgumentError SpinBath([only(s_sites), Index(2)], [sm], sd, OpSum())
    @test_throws ArgumentError BosonicBath(modes=Any[sm], spectral_density=sd, coupling=OpSum())
    @test_throws ArgumentError SpinBath(modes=Any[bm], spectral_density=sd, coupling=OpSum())

    @test_warn r"no mode-system coupling" BosonicBath([bm], sd, OpSum())
    @test_warn r"no mode-system coupling" SpinBath([sm], sd, OpSum())
end

@testset "environments.jl: pretty printing" begin
    sd = ohmic_sd()
    b_sites = liouv_sites(siteinds("Boson", 1; dim=4))
    s_sites = liouv_sites(siteinds("S=1/2", 1))
    n_max_b = dim(only(b_sites)) - 1

    bm = bosonic_mode(
        b_sites,
        OpSum() + (0.1, "N", 1),
        n_max_b,
        random_mps(b_sites);
        coupling=OpSum() + ("N", 1, "Sz", 2),
    )
    sm = spin_mode(
        s_sites,
        OpSum() + (0.1, "Sz", 1),
        random_mps(s_sites);
        coupling=OpSum() + ("Sx", 1, "Sx", 2),
    )
    bb = bosonic_bath([bm, bm])
    sb = spin_bath([sm, sm])

    out_sm = sprint(show, sm)
    out_bm = sprint(show, bm)
    @test out_sm == sprint(show, MIME"text/plain"(), sm)
    @test out_bm == sprint(show, MIME"text/plain"(), bm)

    for (out, name) in ((out_sm, "SpinMode"), (out_bm, "BosonicMode"))
        @test occursin("ProcessTensors.$name", out)
        @test occursin("space: Liouville", out)
        @test occursin("site dims:", out)
        @test occursin("initial state:", out)
        @test occursin("coupling:", out)
    end
    @test occursin("n_max:", out_bm)

    out_sb = sprint(show, sb)
    out_bb = sprint(show, bb)
    @test out_sb == sprint(show, MIME"text/plain"(), sb)
    @test out_bb == sprint(show, MIME"text/plain"(), bb)
    for (out, bath_name, mode_name, dim) in (
        (out_sb, "SpinBath", "SpinMode", 16),
        (out_bb, "BosonicBath", "BosonicMode", 256),
    )
        @test occursin("ProcessTensors.$bath_name", out)
        @test occursin("modes: 2", out)
        @test occursin("space: Liouville", out)
        @test occursin("site dims:", out)
        @test occursin("bath Liouville dimension: $dim", out)
        @test occursin("mode summary:", out)
        @test occursin("[1] $mode_name", out)
        @test occursin("[2] $mode_name", out)
    end
end
