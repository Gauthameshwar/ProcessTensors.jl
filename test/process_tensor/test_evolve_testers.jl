# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: test/process_tensor/test_evolve_testers.jl
# Contributor: Gauthameshwar S.
#
# Tests tester-aware system, tester, and joint trajectories and their
# reduction, timing, and return-shape contracts.
#
# Run with:
# julia --project=. test/runtests.jl

using ProcessTensors
using ITensors
using ITensors.Ops: Trotter
using LinearAlgebra
using Test

if !isdefined(Main, :hilbert_mpo_to_dense)
    include(joinpath(@__DIR__, "..", "time_evolution", "tebd_test_utils.jl"))
end
if !isdefined(Main, :_exact_unitary_exp)
    include(joinpath(@__DIR__, "pt_ed_test_utils.jl"))
end

function _evolve_tester_fixture(; nsteps::Int=3, dt::Real=0.1)
    system_sites = siteinds("Qubit", 1)
    tester_sites = siteinds("Qubit", 1)
    rho0 = to_dm(MPS(system_sites, ["0"]))
    memory = tester(tester_sites, to_dm(MPS(tester_sites, ["1"])))
    pt = build_process_tensor(
        qubit_system(system_sites);
        dt,
        nsteps,
        progress=false,
    )
    return (; pt, system_sites, tester_sites, rho0, memory)
end

function _evolve_density(state::AbstractMPS{Liouville})
    density = to_hilbert(state)
    sites = ProcessTensors._phys_sites_from_hilbert_mpo(density)
    return hilbert_mpo_to_dense(density, sites)
end

function _trace_tester(rho::AbstractMatrix)
    rho4 = reshape(ComplexF64.(rho), 2, 2, 2, 2)
    result = zeros(ComplexF64, 2, 2)
    for a in 1:2
        result .+= @view rho4[:, a, :, a]
    end
    return result
end

function _trace_system(rho::AbstractMatrix)
    rho4 = reshape(ComplexF64.(rho), 2, 2, 2, 2)
    result = zeros(ComplexF64, 2, 2)
    for s in 1:2
        result .+= @view rho4[s, :, s, :]
    end
    return result
end

function _swap_itensor(system_site::Index, tester_site::Index)
    swap = ComplexF64[
        1 0 0 0
        0 0 1 0
        0 1 0 0
        0 0 0 1
    ]
    return ITensor(
        reshape(swap, 2, 2, 2, 2),
        prime(system_site),
        prime(tester_site),
        system_site,
        tester_site,
    )
end

@testset "evolve: tester return shapes and validation" begin
    f = _evolve_tester_fixture()
    baseline = evolve(f.pt, f.rho0; progress=false)
    explicit_no_tester = evolve(
        f.pt,
        f.rho0;
        return_tester=false,
        return_joint=false,
        progress=false,
    )
    @test keys(baseline) == (:times, :states_liouville, :states_hilbert)
    @test keys(explicit_no_tester) == keys(baseline)
    @test baseline.times == [f.pt.dt * k for k in 0:(f.pt.nsteps - 1)]
    @test length(baseline.states_liouville) == f.pt.nsteps
    for k in eachindex(baseline.times)
        @test _evolve_density(explicit_no_tester.states_liouville[k]) ≈
              _evolve_density(baseline.states_liouville[k]) atol=1e-12
    end

    default_tester = evolve(f.pt, f.rho0; tester=f.memory, progress=false)
    @test haskey(default_tester, :tester_states_liouville)
    @test !haskey(default_tester, :joint_states_liouville)

    neither = evolve(
        f.pt,
        f.rho0;
        tester=f.memory,
        return_tester=false,
        return_joint=false,
        progress=false,
    )
    tester_only = default_tester
    joint_only = evolve(
        f.pt,
        f.rho0;
        tester=f.memory,
        return_tester=false,
        return_joint=true,
        progress=false,
    )
    both = evolve(
        f.pt,
        f.rho0;
        tester=f.memory,
        return_tester=true,
        return_joint=true,
        progress=false,
    )
    @test keys(neither) == (:times, :states_liouville, :states_hilbert)
    @test haskey(tester_only, :tester_states_hilbert)
    @test !haskey(tester_only, :joint_states_hilbert)
    @test !haskey(joint_only, :tester_states_hilbert)
    @test haskey(joint_only, :joint_states_hilbert)
    @test haskey(both, :tester_states_hilbert)
    @test haskey(both, :joint_states_hilbert)

    @test_throws ArgumentError evolve(
        f.pt,
        f.rho0;
        return_tester=true,
        progress=false,
    )
    @test_throws ArgumentError evolve(
        f.pt,
        f.rho0;
        return_joint=true,
        progress=false,
    )
    @test_throws ArgumentError evolve(
        f.pt,
        f.rho0;
        tester_seq=TesterSeq(nsteps=f.pt.nsteps),
        progress=false,
    )
    @test_throws ArgumentError evolve(
        f.pt,
        f.rho0;
        tester=f.memory,
        contraction=:evaluate,
        progress=false,
    )
end

@testset "evolve: identity and tester-only controls" begin
    f = _evolve_tester_fixture()
    baseline = evolve(f.pt, f.rho0; progress=false)
    identity_memory = evolve(f.pt, f.rho0; tester=f.memory, progress=false)
    for k in eachindex(baseline.times)
        @test _evolve_density(identity_memory.states_liouville[k]) ≈
              _evolve_density(baseline.states_liouville[k]) atol=1e-11
    end

    controls = TesterSeq(nsteps=f.pt.nsteps)
    add!(
        controls,
        tester_unitary(op("X", only(f.tester_sites)), f.tester_sites),
        1,
    )
    controlled = evolve(
        f.pt,
        f.rho0;
        tester=f.memory,
        tester_seq=controls,
        progress=false,
    )
    @test _evolve_density(controlled.tester_states_liouville[1]) ≈
          ComplexF64[1 0; 0 0] atol=1e-11
    for k in eachindex(baseline.times)
        @test _evolve_density(controlled.states_liouville[k]) ≈
              _evolve_density(baseline.states_liouville[k]) atol=1e-11
        @test tr(_evolve_density(controlled.tester_states_liouville[k])) ≈ 1 atol=1e-11
    end

    propagation = TesterSeq(nsteps=f.pt.nsteps)
    add!(
        propagation,
        tester_propagation(
            OpSum() + (0.7, "X", 1),
            f.tester_sites,
        ),
        1,
    )
    propagated = evolve(
        f.pt,
        f.rho0;
        tester=f.memory,
        tester_seq=propagation,
        progress=false,
    )
    @test !isapprox(
        _evolve_density(propagated.tester_states_liouville[1]),
        ComplexF64[0 0; 0 1];
        atol=1e-5,
    )
end

@testset "evolve: joint SWAP reductions and final evaluation" begin
    f = _evolve_tester_fixture()
    controls = TesterSeq(nsteps=f.pt.nsteps)
    add!(
        controls,
        joint_unitary(
            _swap_itensor(only(f.system_sites), only(f.tester_sites)),
            f.system_sites,
            f.tester_sites,
        ),
        1,
    )
    trajectory = evolve(
        f.pt,
        f.rho0;
        tester=f.memory,
        tester_seq=controls,
        return_joint=true,
        progress=false,
    )

    @test _evolve_density(trajectory.states_liouville[1]) ≈
          ComplexF64[0 0; 0 1] atol=1e-9
    @test _evolve_density(trajectory.tester_states_liouville[1]) ≈
          ComplexF64[1 0; 0 0] atol=1e-9

    for k in eachindex(trajectory.times)
        rho_system = _evolve_density(trajectory.states_liouville[k])
        rho_tester = _evolve_density(trajectory.tester_states_liouville[k])
        rho_joint = _evolve_density(trajectory.joint_states_liouville[k])
        @test _trace_tester(rho_joint) ≈ rho_system atol=1e-9
        @test _trace_system(rho_joint) ≈ rho_tester atol=1e-9
        @test tr(rho_system) ≈ 1 atol=1e-9
        @test tr(rho_tester) ≈ 1 atol=1e-9
        @test tr(rho_joint) ≈ 1 atol=1e-9
    end

    seq = InstrumentSeq(default=identity_operation(), nsteps=f.pt.nsteps)
    add!(seq, state_preparation(f.rho0), 0)
    add!(seq, open_output(), f.pt.nsteps)
    final_evaluation = evaluate_process(
        f.pt,
        seq;
        tester=f.memory,
        tester_seq=controls,
        progress=false,
    )
    @test _evolve_density(trajectory.states_liouville[end]) ≈
          _evolve_density(final_evaluation) atol=1e-9
end

@testset "evolve: multi-site Liouville reconstruction and concrete storage" begin
    one_site = siteinds("Qubit", 1)
    one_liouville_site = liouv_sites(one_site)
    one_state = to_liouville(to_dm(MPS(one_site, ["+"])); sites=one_liouville_site)
    one_reconstructed = ProcessTensors._liouville_mps_from_itensor(
        foldl(*, one_state),
        one_liouville_site,
    )
    @test _evolve_density(one_reconstructed) ≈
          hilbert_mpo_to_dense(to_dm(MPS(one_site, ["+"])), one_site) atol=1e-12

    sites = siteinds("Qubit", 2)
    liouville_sites = liouv_sites(sites)
    psi = MPS(ComplexF64[1, im, -0.5, 0.25im], sites; cutoff=0)
    psi ./= norm(psi)
    state = to_liouville(to_dm(psi); sites=liouville_sites)
    tensor = foldl(*, state)

    reconstructed = ProcessTensors._liouville_mps_from_itensor(
        tensor,
        liouville_sites,
    )
    @test length(reconstructed) == 2
    @test _evolve_density(reconstructed) ≈
          hilbert_mpo_to_dense(to_dm(psi), sites) atol=1e-11

    storage = @inferred ProcessTensors._evolve_storage(Val(true), Val(true), 2)
    @test eltype(storage.states_liouville) == MPS{Liouville}
    @test eltype(storage.states_hilbert) == MPO{Hilbert}
    @test eltype(storage.tester_states_liouville) == MPS{Liouville}
    @test eltype(storage.joint_states_hilbert) == MPO{Hilbert}

    f = _evolve_tester_fixture(nsteps=1)
    verbose_result = evolve(
        f.pt,
        f.rho0;
        tester=f.memory,
        progress=false,
        verbose=true,
    )
    @test length(verbose_result.states_liouville) == 1
end

@testset "evolve: one-mode spin bath vs three-body ED trajectories" begin
    function _partial_traces_sys_bath_tester(rho::AbstractMatrix)
        # hilbert_mpo_to_dense on [sys, bath, tester] is (s', b', a', s, b, a) in column-major.
        r = reshape(ComplexF64.(rho), 2, 2, 2, 2, 2, 2)
        rho_SA = zeros(ComplexF64, 2, 2, 2, 2)
        rho_S = zeros(ComplexF64, 2, 2)
        rho_A = zeros(ComplexF64, 2, 2)
        for bath in 1:2
            rho_SA .+= @view r[:, bath, :, :, bath, :]
        end
        for bath in 1:2, tester in 1:2
            rho_S .+= @view r[:, bath, tester, :, bath, tester]
        end
        for system in 1:2, bath in 1:2
            rho_A .+= @view r[system, bath, :, system, bath, :]
        end
        return reshape(rho_SA, 4, 4), rho_S, rho_A
    end

    dt = 0.05
    nsteps = 4
    system_sites = siteinds("Qubit", 1)
    bath_phys = siteinds("S=1/2", 1)
    tester_sites = siteinds("Qubit", 1)
    memory = tester(tester_sites, to_dm(MPS(tester_sites, ["0"])))
    rho0 = to_dm(MPS(system_sites, ["0"]))

    H_system = OpSum() + (0.4, "Z", 1)
    system = qubit_system(system_sites, H_system)
    bath_liouv = liouv_sites(bath_phys)
    rho_bath = to_liouville(to_dm(MPS(bath_phys, ["Up"])); sites=bath_liouv)
    mode = spin_mode(
        bath_liouv,
        OpSum() + (0.3, "Sx", 1),
        rho_bath;
        coupling=OpSum() + (0.25, "Sz", 1, "Z", 2),
    )
    pt = build_process_tensor(
        system;
        environment=spin_bath([mode]),
        dt,
        nsteps,
        sys_alg=Trotter{2}(),
        progress=false,
    )

    H_control = OpSum()
    H_control += 0.2, "X", 1, "X", 2
    H_control += 0.35, "Z", 2
    controls = TesterSeq(nsteps=nsteps)
    add!(
        controls,
        tester_propagation(OpSum() + (0.45, "X", 1), tester_sites),
        0,
    )
    for k in 1:nsteps
        add!(controls, joint_propagation(H_control, system_sites, tester_sites), k)
    end

    trajectory = evolve(
        pt,
        rho0;
        tester=memory,
        tester_seq=controls,
        return_joint=true,
        progress=false,
    )

    joint_sites = Index[only(system_sites), only(bath_phys), only(tester_sites)]
    H_full = OpSum()
    H_full += 0.4, "Z", 1
    H_full += 0.3, "Sx", 2
    H_full += 0.25, "Z", 1, "Sz", 2
    H_full += 0.2, "X", 1, "X", 3
    H_full += 0.35, "Z", 3
    psi0 = MPS(joint_sites, ["0", "Up", "0"])
    rho0_joint = hilbert_mpo_to_dense(to_dm(psi0), joint_sites)
    U_tester0 = _exact_unitary_exp(
        OpSum() + (0.45, "X", 3),
        joint_sites,
        dt,
    )
    rho_after_init = U_tester0 * rho0_joint * U_tester0'

    for k in 1:nsteps
        U = _exact_unitary_exp(H_full, joint_sites, k * dt)
        exact_SA, exact_S, exact_A = _partial_traces_sys_bath_tester(
            U * rho_after_init * U',
        )
        @test _evolve_density(trajectory.states_liouville[k]) ≈ exact_S atol=1e-3
        @test _evolve_density(trajectory.tester_states_liouville[k]) ≈ exact_A atol=1e-3
        @test _evolve_density(trajectory.joint_states_liouville[k]) ≈ exact_SA atol=1e-3
        @test tr(_evolve_density(trajectory.states_liouville[k])) ≈ 1 atol=1e-8
        @test tr(_evolve_density(trajectory.tester_states_liouville[k])) ≈ 1 atol=1e-8
    end
end
