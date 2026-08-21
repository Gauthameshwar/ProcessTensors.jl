# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: test/process_tensor/test_evolve_contractions.jl
# Contributor: Gauthameshwar S.
#
# Tests that :evaluate and :closures evolve contractions agree on every
# reduced-state snapshot for Dense and ACE process tensors.
#
# Run with:
#   julia --project=. test/runtests.jl

using ProcessTensors
using ITensors
using ITensors.Ops: Trotter
using Test
using LinearAlgebra

if !isdefined(Main, :liouville_state_to_dense)
    include(joinpath(@__DIR__, "..", "time_evolution", "tebd_test_utils.jl"))
end
if !isdefined(Main, :_physical_sites_from_hilbert_mpo)
    include(joinpath(@__DIR__, "pt_ed_test_utils.jl"))
end

function _spin_mode(h_coeff::Real, cpl_coeff::Real; cpl_op::AbstractString="Sz")
    env_phys = siteinds("S=1/2", 1)
    env_liouv = liouv_sites(env_phys)
    rho_env = to_liouville(to_dm(MPS(env_phys, ["Up"])); sites=env_liouv)
    H_mode = OpSum() + (h_coeff, "Sx", 1)
    cpl = OpSum() + (cpl_coeff, cpl_op, 1, cpl_op, 2)
    return spin_mode(env_liouv, H_mode, rho_env; coupling=cpl)
end

function _snapshot_matrices(pt, rho0_or_seq; contraction)
    trajectory = evolve(pt, rho0_or_seq; contraction=contraction, progress=false)
    return [_one_site_liouville_state_to_dense(ρ) for ρ in trajectory.states_liouville]
end

function _assert_contractions_agree(pt, rho0_or_seq; atol=1e-10)
    traj_eval = _snapshot_matrices(pt, rho0_or_seq; contraction=:evaluate)
    traj_close = _snapshot_matrices(pt, rho0_or_seq; contraction=:closures)
    @test length(traj_eval) == pt.nsteps
    @test length(traj_close) == pt.nsteps
    for k in 1:pt.nsteps
        @test traj_eval[k] ≈ traj_close[k] atol=atol
        @test abs(real(tr(traj_close[k])) - 1) < max(atol, 1e-8)
    end
    return traj_eval, traj_close
end

@testset "API surface: evolve contraction names" begin
    @test :EvaluateSnapshots ∉ names(ProcessTensors)
    @test :BondClosures ∉ names(ProcessTensors)
    @test :AbstractEvolveContraction ∉ names(ProcessTensors)
    @test !isdefined(ProcessTensors, :AbstractEvolveContraction)
    @test !isdefined(ProcessTensors, :EvaluateSnapshots)
    @test !isdefined(ProcessTensors, :BondClosures)
    s = siteinds("S=1/2", 1)
    system = spin_system(s, OpSum() + (0.4, "Sz", 1))
    rho0 = to_dm(MPS(s, ["Up"]))
    pt = build_process_tensor(system; dt=0.05, nsteps=2, progress=false)
    traj_default = _snapshot_matrices(pt, rho0; contraction=:closures)
    traj_implicit = [_one_site_liouville_state_to_dense(ρ) for ρ in evolve(pt, rho0; progress=false).states_liouville]
    @test traj_default[1] ≈ traj_implicit[1]
    @test traj_default[2] ≈ traj_implicit[2]
    @test_throws ArgumentError evolve(pt, rho0; contraction=:zipup, progress=false)
end

@testset "evolve contractions: Dense Markovian nsteps=1 and nsteps=4" begin
    s = siteinds("S=1/2", 1)
    system = spin_system(s, OpSum() + (0.4, "Sz", 1) + (0.25, "Sx", 1))
    rho0 = to_dm(MPS(s, ["Up"]))
    pt1 = build_process_tensor(system; dt=0.05, nsteps=1, progress=false)
    _assert_contractions_agree(pt1, rho0)
    pt4 = build_process_tensor(system; dt=0.05, nsteps=4, progress=false)
    _assert_contractions_agree(pt4, rho0)
end

@testset "evolve contractions: Dense single-mode spin bath" begin
    s = siteinds("S=1/2", 1)
    system = spin_system(s, OpSum() + (0.3, "Sz", 1) + (0.2, "Sx", 1))
    bath = spin_bath([_spin_mode(0.5, 0.4)])
    rho0 = to_dm(MPS(s, ["Up"]))
    pt = build_process_tensor(
        system, system.sites[1];
        environment=bath, dt=0.1, nsteps=4, progress=false,
    )
    _assert_contractions_agree(pt, rho0)
end

@testset "evolve contractions: Dense two-mode fused bath" begin
    s = siteinds("S=1/2", 1)
    system = spin_system(s, OpSum() + (0.3, "Sz", 1))
    bath = spin_bath([_spin_mode(0.5, 0.12), _spin_mode(0.35, 0.09; cpl_op="Sx")])
    rho0 = to_dm(MPS(s, ["Up"]))
    pt = build_process_tensor(
        system, system.sites[1];
        environment=bath, dt=0.05, nsteps=3, progress=false,
    )
    _assert_contractions_agree(pt, rho0)
end

@testset "evolve contractions: ACE compressed two-mode PT" begin
    s = siteinds("S=1/2", 1)
    system = spin_system(s, OpSum() + (0.7, "Sx", 1))
    bath = spin_bath([_spin_mode(0.3, 0.2), _spin_mode(0.35, 0.25; cpl_op="Sx")])
    rho0 = to_dm(MPS(s, ["Up"]))
    pt = build_process_tensor(
        system, system.sites[1];
        method=ACE(cutoff=1e-12), environment=bath, dt=0.1, nsteps=5,
        combine_alg=Trotter{2}(), progress=false,
    )
    _assert_contractions_agree(pt, rho0; atol=1e-10)
end

@testset "evolve contractions: mid-schedule unitary agrees on both paths" begin
    s = siteinds("S=1/2", 1)
    system = spin_system(s, OpSum() + (0.5, "Sz", 1) + (0.3, "Sx", 1))
    bath = spin_bath([_spin_mode(0.4, 0.15), _spin_mode(0.45, 0.12; cpl_op="Sx")])
    rho0 = to_dm(MPS(s, ["Up"]))
    pt = build_process_tensor(
        system, system.sites[1];
        method=ACE(cutoff=1e-12), environment=bath, dt=0.1, nsteps=5,
        combine_alg=Trotter{2}(), progress=false,
    )

    seq_id = InstrumentSeq(default=identity_operation(), nsteps=pt.nsteps)
    add!(seq_id, state_preparation(rho0), 0)

    seq_u = InstrumentSeq(default=identity_operation(), nsteps=pt.nsteps)
    add!(seq_u, state_preparation(rho0), 0)
    add!(seq_u, unitary_propagation(system), 2)

    traj_id, _ = _assert_contractions_agree(pt, seq_id; atol=1e-10)
    traj_u, _ = _assert_contractions_agree(pt, seq_u; atol=1e-10)
    @test traj_id[1] ≈ traj_u[1] atol=1e-10
    @test traj_id[2] ≈ traj_u[2] atol=1e-10
    @test norm(traj_id[3] - traj_u[3]) > 1e-8
end
