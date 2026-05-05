using Test
using ITensors
using ProcessTensors

@testset "MPO manipulation forwarding API" begin
    @testset "splitblocks forwarding (spin and boson)" begin
        spin_sites = siteinds("S=1/2", 4)
        boson_sites = siteinds("Boson", 3; dim=3)

        mpo_spin = MPO(spin_sites, "Id")
        mpo_boson = MPO(boson_sites, "Id")

        @test_nowarn splitblocks(mpo_spin)
        @test_nowarn splitblocks(mpo_boson)
        @test_nowarn splitblocks(linkinds, mpo_spin)
        @test_nowarn splitblocks(linkinds, mpo_boson)

        @test splitblocks(mpo_spin) isa MPO{Hilbert}
        @test splitblocks(mpo_boson) isa MPO{Hilbert}
        @test splitblocks(linkinds, mpo_spin) isa MPO{Hilbert}
        @test splitblocks(linkinds, mpo_boson) isa MPO{Hilbert}
    end
end
