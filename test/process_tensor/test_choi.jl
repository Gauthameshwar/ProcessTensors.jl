# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: test/process_tensor/test_choi.jl
# Contributor: John F. Kam
#
# Tests Choi conversion against analytic channels and dense reshuffling.
#
# Run with:
#   julia --project=. test/process_tensor/test_choi.jl

using ProcessTensors
using ITensors
using LinearAlgebra
using Test
import ITensorMPS

if !isdefined(Main, :hilbert_mpo_to_dense)
    include(joinpath(@__DIR__, "..", "time_evolution", "tebd_test_utils.jl"))
end
if !isdefined(Main, :_single_mode_bath)
    include(joinpath(@__DIR__, "pt_ed_test_utils.jl"))
end

# Create a dense Choi reference using array reshaping rather than ITensor combiners.
function _choi_dense_reference(pt::ProcessTensor)
    d = isqrt(dim(only(output_sites(pt, 1))))
    legs = Index[]
    for k in reverse(1:pt.nsteps)
        push!(legs, only(output_sites(pt, k)), only(input_sites(pt, k)))
    end

    A = reshape(Array(contract_core(pt.core), legs...), ntuple(_ -> d, 4 * pt.nsteps))
    permutation = ((1:2:ndims(A))..., (2:2:ndims(A))...)
    D = d^(2 * pt.nsteps)
    return reshape(permutedims(A, permutation), D, D)
end

# Array axes are output-fastest within each input ⊗ output temporal site.
_choi_sites(W) = [only(siteinds(W, j; tags=port, plev=0))
                  for j in reverse(1:length(W)) for port in ("out", "in")]

# Test single-step unitary Choi operator.
@testset "choi_pt_mpo: single-step unitary" begin
    @testset "$family" for family in ("S=1/2", "S=1")
        sites = siteinds(family, 1)
        H = OpSum() + (0.4, "Sx", 1) + (0.7, "Sy", 1) + (0.2, "Sz", 1)
        system = spin_system(sites, H)
        dt = 0.1
        pt = build_process_tensor(system; dt=dt, nsteps=1, progress=false)
        W = choi_pt_mpo(pt)

        U = exp(-im * dt * dense_hamiltonian_matrix(H, sites))
        v = vec(U)
        expected = v * v'
        J = hilbert_mpo_to_dense(W, _choi_sites(W))
        @test J ≈ expected atol=1e-12
    end
end

# Compare a process with memory against an independent dense Choi reference.
# Check output indices, preservation of inner bonds, and that the source is unchanged.
@testset "choi_pt_mpo: memory bonds and dense reference" begin
    H = OpSum() + (0.3, "Sy", 1)
    system = spin_system(siteinds("S=1/2", 1), H)
    bath = _single_mode_bath(only(system.sites))
    pt = build_process_tensor(
        system; environment=bath, method=Dense(), dt=0.1, nsteps=3, progress=false,
    )
    @test maxlinkdim(pt) > 1

    original = deepcopy(pt.core)
    W = choi_pt_mpo(pt)
    @test W isa MPO{Hilbert}
    @test length(W) == pt.nsteps
    @test tr(W) ≈ 2^pt.nsteps atol=1e-12

    for j in 1:pt.nsteps
        input = only(siteinds(W, j; tags="in", plev=0))
        output = only(siteinds(W, j; tags="out", plev=0))
        @test input != output
        @test dim(input) == dim(output) == 2
        @test Set(siteinds(W, j; plev=0)) == Set((input, output))
        @test Set(siteinds(W, j; plev=1)) == Set((prime(input), prime(output)))
        @test all(i -> hastags(i, "Site,Choi,tstep=$j"), (input, output))

        links = Set(i for i in inds(original[j]) if has_tag_token(i, "Link"))
        @test Set(inds(W.core[j])) == union(links, Set((input, output, prime(input), prime(output))))

        @test Set(inds(pt.core[j])) == Set(inds(original[j]))
        @test pt.core[j] ≈ original[j]
    end

    sites = _choi_sites(W)
    J = hilbert_mpo_to_dense(W, sites)
    @test J ≈ _choi_dense_reference(pt) atol=1e-12

    # Apply to an MPS with matching separate input/output legs per core.
    tensors = [prod(onehot(s => 1) for s in siteinds(W, j; plev=0))
               for j in 1:length(W)]
    psi = ProcessTensors.MPS{Hilbert}(ITensorMPS.MPS(tensors))
    phi = apply(W, psi; alg="naive", truncate=false)
    v = hilbert_mps_to_dense(psi, sites)
    @test hilbert_mps_to_dense(phi, sites) ≈ J * v atol=1e-12
end

# The identity process corresponds to a tensor product of Bell operators.
@testset "choi_pt_mpo: identity process" begin
    system = qubit_system(siteinds("Qubit", 1))
    pt = build_process_tensor(system; dt=0.1, nsteps=3, progress=false)
    W = choi_pt_mpo(pt)

    # Unnormalized Bell projector for each identity channel.
    omega = [1.0, 0, 0, 1]
    J_step = omega * omega'
    expected = kron(kron(J_step, J_step), J_step)

    sites = _choi_sites(W)
    @test maxlinkdim(W) == 1
    @test hilbert_mpo_to_dense(W, sites) ≈ expected atol=1e-12
end

@testset "choi_pt_mpo: non-square Liouville dimension" begin
    system = qubit_system(siteinds("Qubit", 1))
    s = Index(3; tags="Liouv,tstep=1")
    core = ITensorMPS.MPO([delta(prime(s), s)])
    pt = ProcessTensor(core, system, nothing, 0.1, 1, only(system.sites))
    @test_throws ArgumentError choi_pt_mpo(pt)
end
