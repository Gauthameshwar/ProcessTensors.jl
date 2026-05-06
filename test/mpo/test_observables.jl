using Test
using ITensors
using ProcessTensors

@testset "MPO observable forwarding API" begin
    @testset "trace forwarding (spin and boson)" begin
        spin_sites = siteinds("S=1/2", 4)
        boson_sites = siteinds("Boson", 3; dim=3)

        mpo_spin = MPO(spin_sites, "Id")
        mpo_boson = MPO(boson_sites, "Id")

        @test_nowarn tr(mpo_spin)
        @test_nowarn tr(mpo_boson)
    end
end
