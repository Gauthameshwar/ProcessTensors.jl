# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: test/process_tensor/test_ace_compress.jl
# Contributor: Gauthameshwar S.
#
# Tests the ACE relative singular-value criterion and the physical stability of
# compressed central-spin process tensors.
#
# Run with:
# julia --project=. test/runtests.jl

using ProcessTensors
using ITensors
using ITensors.Ops: Trotter
using LinearAlgebra
using Test

function _polarized_central_spin_mode(coupling::Real)
    bath_phys = siteinds("S=1/2", 1)
    bath_liouv = liouv_sites(bath_phys)
    rho_bath = to_liouville(
        to_dm(MPS(bath_phys, ["Up"]));
        sites=bath_liouv,
    )
    interaction = OpSum()
    interaction += coupling, "Sx", 1, "Sx", 2
    interaction += coupling, "Sy", 1, "Sy", 2
    interaction += coupling, "Sz", 1, "Sz", 2
    return spin_mode(bath_liouv, OpSum(), rho_bath; coupling=interaction)
end

@testset "ACE relative singular-value criterion" begin
    singular_values = [10.0, 0.11, 0.1, 0.01]
    @test ProcessTensors._ace_relative_keep(
        singular_values; epsilon=1e-2, maxdim=typemax(Int),
    ) == 2
    @test ProcessTensors._ace_relative_keep(
        singular_values; epsilon=0.0, maxdim=typemax(Int),
    ) == 4
    @test ProcessTensors._ace_relative_keep(
        singular_values; epsilon=0.0, maxdim=3,
    ) == 3
end

@testset "ACE polarized central-spin compression stays physical" begin
    system_phys = siteinds("S=1/2", 1)
    system = spin_system(system_phys, OpSum())
    initial_state = to_dm(MPS(system_phys, ["+"]))
    Sx = Array(op("Sx", system_phys[1]), prime(system_phys[1]), system_phys[1])

    ranks = Int[]
    for nmodes in (2, 4, 6)
        bath = spin_bath([
            _polarized_central_spin_mode(1 / nmodes)
            for _ in 1:nmodes
        ])
        pt = build_process_tensor(
            system;
            method=ACE(cutoff=1e-10),
            environment=bath,
            dt=0.1,
            nsteps=8,
            combine_alg=Trotter{2}(),
            progress=false,
        )
        push!(ranks, maxlinkdim(pt.core))

        trajectory = evolve(pt, initial_state; progress=false)
        for rho in trajectory.states_hilbert
            tensor = foldl(*, rho)
            site = only(filter(i -> plev(i) == 0 && hastags(i, "Site"), inds(tensor)))
            matrix = Array(tensor, prime(site), site)
            trace_value = tr(matrix)
            expectation = real(tr(Sx * matrix) / trace_value)
            @test abs(trace_value - 1) < 3e-5
            @test abs(expectation) <= 0.5 + 1e-8
        end
    end

    # Relative SVD must not collapse below the four-dimensional effective
    # spin memory of the polarized model.
    @test all(>=(4), ranks)
end

@testset "ACE Hilbert unitary channel matches Liouville exp" begin
    phys = siteinds("S=1/2", 2)
    liouv = liouv_sites(phys)
    os = OpSum()
    os += 0.3, "Sz", 1
    os += 0.2, "Sx", 1, "Sx", 2
    os += 0.2, "Sy", 1, "Sy", 2
    os += 0.2, "Sz", 1, "Sz", 2
    U_h = ProcessTensors._hilbert_unitary_liouville_propagator(os, liouv, 0.01)
    U_l = liouvillian_propagator(os, liouv, 0.01)
    @test norm(U_h - U_l) / norm(U_l) < 1e-12
end
