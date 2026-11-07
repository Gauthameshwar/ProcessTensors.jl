# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: test/measures/test_temporal_entanglement.jl
# Contributor: Gauthameshwar S.
#
# Tests that temporal_entanglement returns the normalized Schmidt entropy of an MPO
# or process tensor, including a full bond profile without a second QR sweep.
#
# Run with:
#   julia --project=. test/runtests.jl

using Test
using ITensors
using ProcessTensors

@testset "API surface: temporal entanglement names" begin
    @test :temporal_entanglement ∈ names(ProcessTensors)
end

# Squared Schmidt weights from a matricization, normalized the same way as temporal_entanglement.
function _reference_entropy(tensor::ITensor, left_inds)
    _, _, _, spec = svd(tensor, left_inds...; cutoff=0.0)
    w = eigs(spec)
    z = sum(w)
    z > 0 || return 0.0
    S = 0.0
    for wi in w
        p = wi / z
        p > 0 && (S -= p * log(p))
    end
    return S
end

# Site indices of one MPO core, excluding the virtual bonds shared with its neighbors.
function _core_siteinds(cores, k)
    kept = Index[]
    for i in inds(cores[k])
        shared_left = k > 1 && i in commoninds(cores[k], cores[k - 1])
        shared_right = k < length(cores) && i in commoninds(cores[k], cores[k + 1])
        (shared_left || shared_right) || push!(kept, i)
    end
    return kept
end

function _contracted_cut_entropy(pt::ProcessTensor, b::Integer)
    cores = [copy(pt[k]) for k in 1:length(pt)]
    tensor = foldl(*, cores)
    left = Index[]
    for k in 1:b
        append!(left, _core_siteinds(cores, k))
    end
    return _reference_entropy(tensor, left)
end

function _bath_process(; nsteps::Integer)
    sites = siteinds("S=1/2", 1)
    H = OpSum() + (0.3, "Sx", 1) + (0.2, "Sz", 1)
    system = spin_system(sites, H)
    env_phys = siteinds("S=1/2", 1)
    env_liouv = liouv_sites(env_phys)
    rho_env = to_liouville(to_dm(MPS(env_phys, ["Up"])); sites=env_liouv)
    mode = spin_mode(
        env_liouv,
        OpSum() + (1.0, "Sx", 1),
        rho_env;
        coupling=OpSum() + (1.5, "Sz", 1, "Sz", 2),
    )
    pt = build_process_tensor(
        system,
        system.sites[1];
        environment=spin_bath([mode]),
        dt=0.4,
        nsteps=nsteps,
        progress=false,
    )
    return pt
end

function _equal_schmidt_mpo()
    sites = siteinds("S=1/2", 2)
    link = Index(2; tags="Link,l=1")
    left = ITensor(sites[1], prime(sites[1]), link)
    left[sites[1] => 1, prime(sites[1]) => 1, link => 1] = 1.0
    left[sites[1] => 2, prime(sites[1]) => 2, link => 2] = 1.0
    right = ITensor(dag(link), sites[2], prime(sites[2]))
    amplitude = 1 / sqrt(2)
    right[link => 1, sites[2] => 1, prime(sites[2]) => 1] = amplitude
    right[link => 2, sites[2] => 2, prime(sites[2]) => 2] = amplitude
    return MPO([left, right])
end

@testset "temporal_entanglement" begin
    @testset "product process tensor" begin
        sites = siteinds("S=1/2", 1)
        system = spin_system(sites, OpSum() + (0.3, "Sx", 1))
        pt = build_process_tensor(system; dt=0.05, nsteps=4, progress=false)
        profile = temporal_entanglement(pt)
        @test length(profile) == 3
        @test all(S -> isapprox(S, 0.0; atol=1e-10), profile)

        pt[2] = 3 * pt[2]
        @test all(S -> isapprox(S, 0.0; atol=1e-10), temporal_entanglement(pt))
    end

    @testset "one-step process and invalid cuts" begin
        sites = siteinds("S=1/2", 1)
        system = spin_system(sites, OpSum() + (0.3, "Sz", 1))
        pt = build_process_tensor(system; dt=0.05, nsteps=1, progress=false)
        @test temporal_entanglement(pt) == Float64[]
        @test_throws ArgumentError temporal_entanglement(pt, 0)
        @test_throws ArgumentError temporal_entanglement(pt, 1)

        longer = build_process_tensor(system; dt=0.05, nsteps=3, progress=false)
        @test_throws ArgumentError temporal_entanglement(longer, 0)
        @test_throws ArgumentError temporal_entanglement(longer, longer.nsteps)
    end

    @testset "equal Schmidt MPO" begin
        W = _equal_schmidt_mpo()
        @test temporal_entanglement(W, 1) ≈ log(2)
        W[1] = 4 * W[1]
        @test temporal_entanglement(W, 1) ≈ log(2)
        @test only(temporal_entanglement(W)) ≈ log(2)
        @test temporal_entanglement(W, 1) ≈ log(2)
    end

    @testset "dense reference and profile agreement" begin
        pt_ref = _bath_process(; nsteps=3)
        reference = [_contracted_cut_entropy(pt_ref, b) for b in 1:2]
        @test temporal_entanglement(pt_ref) ≈ reference atol = 1e-8

        pt_profile = _bath_process(; nsteps=3)
        pt_scalar = _bath_process(; nsteps=3)
        profile = temporal_entanglement(pt_profile)
        scalars = [temporal_entanglement(pt_scalar, b) for b in 1:2]
        @test profile ≈ scalars atol = 1e-8
        for b in 1:2
            @test temporal_entanglement(pt_profile, b) ≈ profile[b] atol = 1e-8
        end
        @test any(S -> S > 1e-3, profile)
    end

    @testset "gauge leaves the process unchanged" begin
        sites = siteinds("S=1/2", 1)
        system = spin_system(sites, OpSum() + (0.3, "Sx", 1) + (0.2, "Sz", 1))
        pt = build_process_tensor(system; dt=0.05, nsteps=4, progress=false)
        rho0 = to_dm(MPS(sites, ["Up"]))
        before = evolve(pt, rho0; progress=false)
        hs_norm = norm(pt)
        temporal_entanglement(pt)
        after = evolve(pt, rho0; progress=false)
        @test norm(pt) ≈ hs_norm
        for (left, right) in zip(before.states_liouville, after.states_liouville)
            @test inner(left, right) ≈ inner(left, left)
        end
    end
end
