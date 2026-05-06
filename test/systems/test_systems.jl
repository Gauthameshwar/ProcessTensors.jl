using ProcessTensors
using ITensors
using Test

@testset "systems.jl: type and field definitions" begin
    @test SpinSystem <: AbstractSystem
    @test BosonSystem <: AbstractSystem

    @test hasfield(SpinSystem, :H)
    @test hasfield(SpinSystem, :jump_ops)
    @test hasfield(SpinSystem, :sites)
    @test fieldtype(SpinSystem, :H) == OpSum
    @test fieldtype(SpinSystem, :jump_ops) == Vector{OpSum}
    @test fieldtype(SpinSystem, :sites) == Vector{Index}

    @test hasfield(BosonSystem, :H)
    @test hasfield(BosonSystem, :jump_ops)
    @test hasfield(BosonSystem, :sites)
    @test fieldtype(BosonSystem, :H) == OpSum
    @test fieldtype(BosonSystem, :jump_ops) == Vector{OpSum}
    @test fieldtype(BosonSystem, :sites) == Vector{Index}
end

@testset "systems.jl: single-site spin and boson construction" begin
    s_spin = siteinds("S=1/2", 1)
    s_boson = siteinds("Boson", 1; dim=3)

    H_spin = OpSum() + (0.4, "Sz", 1)
    H_boson = OpSum() + (0.2, "N", 1)
    L_spin = OpSum() + (0.1, "S-", 1)
    L_boson = OpSum() + (0.1, "A", 1)

    sys_spin = spin_system(s_spin, H_spin; jump_ops=[L_spin])
    sys_boson = boson_system(s_boson, H_boson; jump_ops=[L_boson])

    @test sys_spin isa SpinSystem
    @test sys_boson isa BosonSystem
    @test length(sys_spin.sites) == 1
    @test length(sys_boson.sites) == 1
    @test all(has_tag_token(s, "Liouv") for s in sys_spin.sites)
    @test all(has_tag_token(s, "Liouv") for s in sys_boson.sites)
    @test eltype(sys_spin.jump_ops) == OpSum
    @test eltype(sys_boson.jump_ops) == OpSum
    @test eltype(sys_spin.sites) == Index
    @test eltype(sys_boson.sites) == Index
end

@testset "systems.jl: multi-site spin and boson construction" begin
    s_spin = siteinds("S=1/2", 3)
    s_boson = siteinds("Boson", 3; dim=3)

    H_spin = OpSum() + (0.4, "Sz", 1) + (0.2, "Sx", 2) + (0.1, "Sz", 3)
    H_boson = OpSum() + (0.2, "N", 1) + (0.3, "N", 2) + (0.4, "N", 3)
    L_spin = OpSum() + (0.05, "S-", 2)
    L_boson = OpSum() + (0.05, "A", 2)

    @test_nowarn spin_system(s_spin, H_spin; jump_ops=[L_spin])
    @test_nowarn boson_system(s_boson, H_boson; jump_ops=[L_boson])

    sys_spin = spin_system(s_spin, H_spin; jump_ops=[L_spin])
    sys_boson = boson_system(s_boson, H_boson; jump_ops=[L_boson])

    @test length(sys_spin.sites) == 3
    @test length(sys_boson.sites) == 3
    @test all(has_tag_token(s, "Liouv") for s in sys_spin.sites)
    @test all(has_tag_token(s, "Liouv") for s in sys_boson.sites)
end

@testset "systems.jl: direct Liouville-site inputs" begin
    spin_liouv = liouv_sites(siteinds("S=1/2", 2))
    boson_liouv = liouv_sites(siteinds("Boson", 2; dim=3))

    H_spin = OpSum() + (0.3, "Sz", 1)
    H_boson = OpSum() + (0.2, "N", 1)

    sys_spin = spin_system(spin_liouv, H_spin)
    sys_boson = boson_system(boson_liouv, H_boson)

    @test length(sys_spin.sites) == 2
    @test length(sys_boson.sites) == 2
    @test all(has_tag_token(s, "Liouv") for s in sys_spin.sites)
    @test all(has_tag_token(s, "Liouv") for s in sys_boson.sites)
end

@testset "systems.jl: mixed Hilbert/Liouville inputs reject" begin
    spin_h = siteinds("S=1/2", 2)
    spin_mix = [spin_h[1], liouv_sites(spin_h)[2]]
    boson_h = siteinds("Boson", 2; dim=3)
    boson_mix = [boson_h[1], liouv_sites(boson_h)[2]]

    H_spin = OpSum() + (0.2, "Sz", 1)
    H_boson = OpSum() + (0.2, "N", 1)

    @test_throws ArgumentError spin_system(spin_mix, H_spin)
    @test_throws ArgumentError boson_system(boson_mix, H_boson)
end

@testset "systems.jl: wrong site family rejects" begin
    spin_sites = siteinds("S=1/2", 2)
    boson_sites = siteinds("Boson", 2; dim=3)

    H_spin = OpSum() + (0.2, "Sz", 1)
    H_boson = OpSum() + (0.2, "N", 1)

    @test_throws ArgumentError spin_system(boson_sites, H_spin)
    @test_throws ArgumentError boson_system(spin_sites, H_boson)
end

@testset "systems.jl: empty Hamiltonian warning path" begin
    spin_sites = siteinds("S=1/2", 1)
    boson_sites = siteinds("Boson", 1; dim=3)

    @test_warn r"SpinSystem: H is empty" spin_system(spin_sites, OpSum())
    @test_warn r"BosonSystem: H is empty" boson_system(boson_sites, OpSum())
end

@testset "systems.jl: functional constructor defaults and explicit jump_ops" begin
    spin_sites = siteinds("S=1/2", 2)
    boson_sites = siteinds("Boson", 2; dim=3)
    H_spin = OpSum() + (0.3, "Sz", 1)
    H_boson = OpSum() + (0.4, "N", 1)
    J_spin = OpSum() + (0.1, "S-", 1)
    J_boson = OpSum() + (0.1, "A", 1)

    sys_spin_default = spin_system(spin_sites, H_spin)
    sys_boson_default = boson_system(boson_sites, H_boson)
    sys_spin_explicit = spin_system(spin_sites, H_spin; jump_ops=[J_spin])
    sys_boson_explicit = boson_system(boson_sites, H_boson; jump_ops=[J_boson])

    @test isempty(sys_spin_default.jump_ops)
    @test isempty(sys_boson_default.jump_ops)
    @test length(sys_spin_explicit.jump_ops) == 1
    @test length(sys_boson_explicit.jump_ops) == 1
end
