# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: test/environments/test_environments.jl
# Contributor: Gauthameshwar S.
#
# Tests bath and bath-mode constructors, fields, and environment API contracts.
#
# Run with:
#   julia --project=. test/runtests.jl

using ProcessTensors
using ProcessTensors.Spectrals: ohmic_sd
using ITensors
using LinearAlgebra: I, Diagonal
using Test

@testset "API surface: bath names and fields" begin
    for (type_name, user_ctor_name) in (
        (:BosonicMode, :bosonic_mode),
        (:SpinMode, :spin_mode),
        (:BosonicBath, :bosonic_bath),
        (:SpinBath, :spin_bath),
    )
        @test type_name ∈ names(ProcessTensors)
        @test user_ctor_name ∈ names(ProcessTensors)
    end
    @test nameof(BosonicMode) == :BosonicMode
    @test :mode_initial_states ∈ names(ProcessTensors)
    @test :thermal_mode ∈ names(ProcessTensors)

    b_sites = liouv_sites(siteinds("Boson", 1; dim=4))
    s_sites = liouv_sites(siteinds("S=1/2", 1))
    rho_b = random_mps(b_sites)
    rho_s = random_mps(s_sites)
    H_b = OpSum() + (0.3, "N", 1)
    H_s = OpSum() + (0.5, "Sz", 1)
    coupling_b = OpSum() + (0.1, "N", 1, "Sz", 2)
    coupling_s = OpSum() + (0.2, "Sz", 1, "Sz", 2)

    mode_b = bosonic_mode(b_sites, H_b, rho_b; coupling=coupling_b)
    mode_s = spin_mode(s_sites, H_s, rho_s; coupling=coupling_s)
    thermal_b = thermal_mode(b_sites, H_b, 1.0; coupling=coupling_b)
    thermal_s = thermal_mode(s_sites, H_s, 1.0; coupling=coupling_s)
    @test nameof(typeof(mode_b)) == :BosonicMode
    @test nameof(typeof(mode_s)) == :SpinMode
    @test nameof(typeof(thermal_b)) == :BosonicMode
    @test nameof(typeof(thermal_s)) == :SpinMode
    @test nameof(typeof(thermal_mode(mode_b, 1.0))) == :BosonicMode
    @test mode_b.sites == b_sites
    @test mode_b.H == H_b
    @test mode_b.coupling == coupling_b
    @test mode_b.rho0 == rho_b
    @test mode_s.sites == s_sites
    @test mode_s.H == H_s
    @test mode_s.coupling == coupling_s
    @test mode_s.rho0 == rho_s

    sd = ohmic_sd()
    coupling_bath = OpSum()
    bath_b = bosonic_bath([mode_b]; spectral_density=sd, coupling=coupling_bath)
    bath_s = spin_bath([mode_s]; spectral_density=sd, coupling=coupling_bath)
    @test nameof(typeof(bath_b)) == :BosonicBath
    @test nameof(typeof(bath_s)) == :SpinBath
    @test bath_b.modes == [mode_b]
    @test bath_b.coupling == coupling_bath
    @test bath_b.spectral_density == sd
    @test mode_initial_states(bath_b) == getfield.(bath_b.modes, :rho0)
    @test bath_s.modes == [mode_s]
    @test bath_s.coupling == coupling_bath
    @test bath_s.spectral_density == sd
    @test mode_initial_states(bath_s) == getfield.(bath_s.modes, :rho0)
end

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

@testset "environments.jl: strict mode constructors" begin
    b_sites = liouv_sites(siteinds("Boson", 1; dim=4))
    s_sites = liouv_sites(siteinds("S=1/2", 1))
    rho_b = random_mps(b_sites)
    rho_s = random_mps(s_sites)
    n_max_b = dim(only(b_sites)) - 1

    H_b = OpSum() + (0.3, "N", 1)
    H_s = OpSum() + (0.5, "Sz", 1)

    bm = @test_nowarn BosonicMode(b_sites, H_b, n_max_b, rho_b)
    sm = @test_nowarn SpinMode(s_sites, H_s, rho_s)
    @test bm isa BosonicMode
    @test sm isa SpinMode
    @test bm.n_max == n_max_b
    @test length(bm.sites) == 1
    @test length(sm.sites) == 1
end

@testset "environments.jl: mode user-facing constructors" begin
    b_sites = liouv_sites(siteinds("Boson", 1; dim=4))
    s_sites = liouv_sites(siteinds("S=1/2", 1))
    rho_b = random_mps(b_sites)
    rho_s = random_mps(s_sites)
    n_max_b = dim(only(b_sites)) - 1
    H_b = OpSum() + (0.3, "N", 1)
    H_s = OpSum() + (0.5, "Sz", 1)

    bm1 = bosonic_mode(b_sites, H_b, n_max_b, rho_b)
    bm2 = bosonic_mode(b_sites, H_b, rho_b)
    bm3 = bosonic_mode(; sites=b_sites, H=H_b, rho0=rho_b)
    sm1 = spin_mode(s_sites, H_s, rho_s)
    sm2 = spin_mode(; sites=s_sites, H=H_s, rho0=rho_s)

    @test bm1 isa BosonicMode
    @test bm2.n_max == n_max_b
    @test bm3.n_max == n_max_b
    @test sm1 isa SpinMode
    @test sm2.sites == sm1.sites
    @test bm1.H == bm2.H
    @test bm1.rho0 == bm2.rho0
end

@testset "environments.jl: removed uppercase mode helpers" begin
    b_sites = liouv_sites(siteinds("Boson", 1; dim=4))
    s_sites = liouv_sites(siteinds("S=1/2", 1))
    rho_b = random_mps(b_sites)
    rho_s = random_mps(s_sites)
    H_b = OpSum() + (0.3, "N", 1)
    H_s = OpSum() + (0.5, "Sz", 1)

    @test_throws MethodError BosonicMode(b_sites, H_b, rho_b)
    @test_throws MethodError BosonicMode(; sites=b_sites, H=H_b, rho0=rho_b)
    @test_throws MethodError SpinMode(; sites=s_sites, H=H_s, rho0=rho_s)
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

@testset "environments.jl: bath constructors and user-facing constructors" begin
    sd = ohmic_sd()

    b1_sites = liouv_sites(siteinds("Boson", 1; dim=4))
    b2_sites = liouv_sites(siteinds("Boson", 1; dim=5))
    s1_sites = liouv_sites(siteinds("S=1/2", 1))
    s2_sites = liouv_sites(siteinds("S=1/2", 1))

    b1 = bosonic_mode(b1_sites, OpSum() + (0.1, "N", 1), dim(only(b1_sites)) - 1, random_mps(b1_sites); coupling=OpSum() + (0.2, "N", 1, "Sz", 2))
    b2 = bosonic_mode(b2_sites, OpSum() + (0.1, "N", 1), dim(only(b2_sites)) - 1, random_mps(b2_sites); coupling=OpSum() + (0.2, "N", 1, "Sz", 2))
    s1 = spin_mode(s1_sites, OpSum() + (0.1, "Sz", 1), random_mps(s1_sites); coupling=OpSum() + (0.2, "Sz", 1, "Sz", 2))
    s2 = spin_mode(s2_sites, OpSum() + (0.1, "Sz", 1), random_mps(s2_sites); coupling=OpSum() + (0.2, "Sz", 1, "Sz", 2))

    sites_b = collect(Iterators.flatten(getfield.([b1, b2], :sites)))
    sites_s = collect(Iterators.flatten(getfield.([s1, s2], :sites)))

    bb_strict = @test_nowarn BosonicBath(sites_b, [b1, b2], sd, OpSum())
    sb_strict = @test_nowarn SpinBath(sites_s, [s1, s2], sd, OpSum())
    @test bb_strict isa BosonicBath
    @test sb_strict isa SpinBath

    bb = @test_nowarn bosonic_bath([b1, b2]; spectral_density=sd)
    sb = @test_nowarn spin_bath([s1, s2]; spectral_density=sd)
    bb_kw = @test_nowarn bosonic_bath(; modes=[b1, b2], spectral_density=sd)
    sb_kw = @test_nowarn spin_bath(; modes=[s1, s2], spectral_density=sd)

    @test length(bb.modes) == 2
    @test length(sb.modes) == 2
    @test length(bb_kw.modes) == 2
    @test length(sb_kw.modes) == 2
    @test length(mode_initial_states(bb)) == 2
    @test length(mode_initial_states(sb)) == 2
    @test bb.sites == bb_strict.sites
    @test sb.sites == sb_strict.sites
end

@testset "environments.jl: removed uppercase bath helpers" begin
    sd = ohmic_sd()
    b_sites = liouv_sites(siteinds("Boson", 1; dim=4))
    s_sites = liouv_sites(siteinds("S=1/2", 1))
    bm = bosonic_mode(b_sites, OpSum() + (0.1, "N", 1), dim(only(b_sites)) - 1, random_mps(b_sites); coupling=OpSum() + (0.2, "N", 1, "Sz", 2))
    sm = spin_mode(s_sites, OpSum() + (0.1, "Sz", 1), random_mps(s_sites); coupling=OpSum() + (0.2, "Sz", 1, "Sz", 2))

    @test_throws MethodError BosonicBath([bm], sd)
    @test_throws MethodError SpinBath([sm], sd)
    @test_throws MethodError BosonicBath(; modes=[bm], spectral_density=sd)
    @test_throws MethodError SpinBath(; modes=[sm], spectral_density=sd)
end

@testset "environments.jl: bath validation errors/warnings" begin
    sd = ohmic_sd()
    b_sites = liouv_sites(siteinds("Boson", 1; dim=4))
    s_sites = liouv_sites(siteinds("S=1/2", 1))
    bm = bosonic_mode(b_sites, OpSum() + (0.1, "N", 1), dim(only(b_sites)) - 1, random_mps(b_sites))
    sm = spin_mode(s_sites, OpSum() + (0.1, "Sz", 1), random_mps(s_sites))

    @test_throws ArgumentError BosonicBath([only(b_sites), Index(2)], [bm], sd, OpSum())
    @test_throws ArgumentError SpinBath([only(s_sites), Index(2)], [sm], sd, OpSum())
    @test_throws ArgumentError bosonic_bath(; modes=Any[sm], spectral_density=sd, coupling=OpSum())
    @test_throws ArgumentError spin_bath(; modes=Any[bm], spectral_density=sd, coupling=OpSum())

    @test_warn r"no mode-system coupling" bosonic_bath([bm]; spectral_density=sd, coupling=OpSum())
    @test_warn r"no mode-system coupling" spin_bath([sm]; spectral_density=sd, coupling=OpSum())
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

@testset "environments.jl: thermal_mode Gibbs and limits" begin
    function dense_mode_density(mode)
        ρh = to_hilbert(mode.rho0)
        tensor = foldl(*, ρh)
        phys = only(ProcessTensors._phys_sites_from_hilbert_mpo(ρh))
        return ComplexF64.(Array(tensor, prime(phys), phys))
    end

    ω = 0.7
    T = 1.3
    n_phys = 4
    b_sites = liouv_sites(siteinds("Boson", 1; dim=n_phys))
    s_sites = liouv_sites(siteinds("S=1/2", 1))
    H_b = OpSum() + (ω, "N", 1)
    H_s = OpSum() + (ω, "Sz", 1)
    coupling_b = OpSum() + (0.1, "N", 1, "Sz", 2)
    coupling_s = OpSum() + (0.2, "Sz", 1, "Sz", 2)

    bm = thermal_mode(b_sites, H_b, T; coupling=coupling_b)
    sm = thermal_mode(s_sites, H_s, T; coupling=coupling_s)
    @test bm isa BosonicMode
    @test sm isa SpinMode
    @test bm.H == H_b
    @test bm.coupling == coupling_b
    @test bm.sites == b_sites
    @test bm.n_max == dim(only(b_sites)) - 1
    @test sm.H == H_s
    @test sm.coupling == coupling_s
    @test sm.sites == s_sites

    occupations = 0:(n_phys - 1)
    boson_weights = exp.(-ω .* occupations ./ T)
    boson_weights ./= sum(boson_weights)
    ρb = dense_mode_density(bm)
    @test ρb ≈ Diagonal(boson_weights) atol=1e-10

    spin_energies = [ω / 2, -ω / 2]
    spin_weights = exp.(-spin_energies ./ T)
    spin_weights ./= sum(spin_weights)
    ρs = dense_mode_density(sm)
    @test ρs ≈ Diagonal(spin_weights) atol=1e-10

    seed = bosonic_mode(b_sites, H_b, random_mps(b_sites); coupling=coupling_b)
    replaced = thermal_mode(seed, T)
    @test replaced.H == seed.H
    @test replaced.coupling == seed.coupling
    @test replaced.sites == seed.sites
    @test replaced.n_max == seed.n_max
    @test dense_mode_density(replaced) ≈ ρb atol=1e-10

    ρ0 = dense_mode_density(thermal_mode(b_sites, H_b, 0.0; coupling=coupling_b))
    vacuum = zeros(ComplexF64, n_phys, n_phys)
    vacuum[1, 1] = 1
    @test ρ0 ≈ vacuum atol=1e-10

    ρ∞ = dense_mode_density(thermal_mode(b_sites, H_b, Inf; coupling=coupling_b))
    @test ρ∞ ≈ Matrix{ComplexF64}(I, n_phys, n_phys) / n_phys atol=1e-12

    ρ_empty = @test_warn r"BosonicMode:H is empty" dense_mode_density(
        thermal_mode(b_sites, OpSum(), 0.0),
    )
    @test ρ_empty ≈ Matrix{ComplexF64}(I, n_phys, n_phys) / n_phys atol=1e-12

    @test_throws DomainError thermal_mode(b_sites, H_b, -1.0)
    @test_throws DomainError thermal_mode(bm, NaN)
end
