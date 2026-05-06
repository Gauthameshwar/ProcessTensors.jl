using Test
using ITensors
using ProcessTensors

@testset "MPO constructor forwarding API" begin
    @testset "random_mpo forwarding (spin and boson)" begin
        spin_sites = siteinds("S=1/2", 4)
        boson_sites = siteinds("Boson", 3; dim=3)

        # random_mpo(sites; kwargs...)
        @test_nowarn random_mpo(spin_sites)
        @test_nowarn random_mpo(boson_sites)
        @test random_mpo(spin_sites) isa MPO{Hilbert}
        @test random_mpo(boson_sites) isa MPO{Hilbert}

    end
end
