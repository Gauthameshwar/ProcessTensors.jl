# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: test/process_tensor/test_evaluate_testers.jl
# Contributor: Gauthameshwar S.
#
# Tests tester-aware process evaluation against exact one-system/one-ancilla
# references and schedule-validation contracts.
#
# Run with:
# julia --project=. test/runtests.jl

using ProcessTensors
using ITensors
using ITensors.Ops: Trotter
using LinearAlgebra
using Test

if !isdefined(Main, :dense_hamiltonian_matrix)
    include(joinpath(@__DIR__, "..", "time_evolution", "tebd_test_utils.jl"))
end
if !isdefined(Main, :_partial_trace_env)
    include(joinpath(@__DIR__, "pt_ed_test_utils.jl"))
end

const TesterCompile = ProcessTensors.Instruments

function _tester_result_dense(result::AbstractMPS{Liouville})
    density = to_hilbert(result)
    site = only(ProcessTensors._phys_sites_from_hilbert_mpo(density))
    return reshape(
        ComplexF64.(Array(foldl(*, density), prime(site), site)),
        dim(site),
        dim(site),
    )
end

function _tester_schedule(pt, rho0; open::Bool)
    seq = InstrumentSeq(default=identity_operation(), nsteps=pt.nsteps)
    add!(seq, state_preparation(rho0), 0)
    add!(seq, open ? open_output() : trace_out(), pt.nsteps)
    return seq
end

function _qubit_tester()
    sites = siteinds("Qubit", 1)
    return sites, tester(sites, to_dm(MPS(sites, ["0"])))
end

@testset "evaluate_process: tester memory and isolated actions" begin
    system_sites = siteinds("Qubit", 1)
    rho0 = to_dm(MPS(system_sites, ["0"]))
    pt = build_process_tensor(
        qubit_system(system_sites);
        dt=0.1,
        nsteps=3,
        progress=false,
    )
    seq_open = _tester_schedule(pt, rho0; open=true)
    seq_closed = _tester_schedule(pt, rho0; open=false)
    tester_sites, memory = _qubit_tester()

    baseline_open = _tester_result_dense(evaluate_process(pt, seq_open; progress=false))
    baseline_closed = evaluate_process(pt, seq_closed; progress=false)

    identity_open = evaluate_process(pt, seq_open; tester=memory, progress=false)
    identity_closed = evaluate_process(pt, seq_closed; tester=memory, progress=false)
    @test _tester_result_dense(identity_open) ≈ baseline_open atol=1e-11
    @test identity_closed ≈ baseline_closed atol=1e-11

    unitary_seq = TesterSeq(nsteps=pt.nsteps)
    add!(unitary_seq, tester_unitary(op("X", only(tester_sites)), tester_sites), 1)
    isolated_unitary = evaluate_process(
        pt,
        seq_open;
        tester=memory,
        tester_seq=unitary_seq,
        progress=false,
    )
    @test _tester_result_dense(isolated_unitary) ≈ baseline_open atol=1e-11

    propagation_seq = TesterSeq(nsteps=pt.nsteps)
    H_tester = OpSum() + (0.7, "X", 1)
    add!(propagation_seq, tester_propagation(H_tester, tester_sites), 2)
    isolated_propagation = evaluate_process(
        pt,
        seq_open;
        tester=memory,
        tester_seq=propagation_seq,
        progress=false,
    )
    @test _tester_result_dense(isolated_propagation) ≈ baseline_open atol=1e-11

    @test evaluate_process(
        pt,
        [seq_closed, seq_closed];
        tester=memory,
        progress=false,
    ) ≈ fill(baseline_closed, 2)
    @test evaluate_process(
        pt,
        rho0,
        InstrumentSeq(default=identity_operation(), nsteps=pt.nsteps);
        tester=memory,
        progress=false,
    ) isa MPS{Liouville}
end

@testset "evaluate_process: joint Hamiltonian reference" begin
    system_sites = siteinds("Qubit", 1)
    tester_sites, memory = _qubit_tester()
    dt = 0.2
    coupling = 0.8
    pt = build_process_tensor(
        qubit_system(system_sites);
        dt,
        nsteps=1,
        progress=false,
    )
    rho0 = to_dm(MPS(system_sites, ["0"]))
    seq = _tester_schedule(pt, rho0; open=true)

    zero_seq = TesterSeq(nsteps=1)
    add!(
        zero_seq,
        joint_propagation(OpSum(), system_sites, tester_sites),
        1,
    )
    zero_result = evaluate_process(
        pt,
        seq;
        tester=memory,
        tester_seq=zero_seq,
        progress=false,
    )
    @test _tester_result_dense(zero_result) ≈ ComplexF64[1 0; 0 0] atol=1e-11

    H_joint = OpSum() + (coupling, "X", 1, "X", 2)
    joint_seq = TesterSeq(nsteps=1)
    add!(joint_seq, joint_propagation(H_joint, system_sites, tester_sites), 1)
    result = evaluate_process(
        pt,
        seq;
        tester=memory,
        tester_seq=joint_seq,
        progress=false,
    )
    expected = ComplexF64[
        cos(coupling * dt)^2 0
        0 sin(coupling * dt)^2
    ]
    @test _tester_result_dense(result) ≈ expected atol=1e-10
end

@testset "evaluate_process: joint unitary square root and CNOT" begin
    system_sites = siteinds("Qubit", 1)
    tester_sites, memory = _qubit_tester()
    U = op("CNOT", only(system_sites), only(tester_sites))
    physical_sites = vcat(system_sites, tester_sites)
    U_half = TesterCompile._unitary_half(U, physical_sites)
    U_matrix = TesterCompile._unitary_matrix(U, physical_sites; owner="test")
    U_half_matrix = TesterCompile._unitary_matrix(
        U_half,
        physical_sites;
        owner="test",
    )
    @test U_half_matrix * U_half_matrix ≈ U_matrix atol=1e-9
    @test U_half_matrix' * U_half_matrix ≈ I atol=1e-10

    pt = build_process_tensor(
        qubit_system(system_sites);
        dt=0.1,
        nsteps=1,
        progress=false,
    )
    rho_plus = to_dm(MPS(system_sites, ["+"]))
    seq = _tester_schedule(pt, rho_plus; open=true)
    controls = TesterSeq(nsteps=1)
    add!(controls, joint_unitary(U, system_sites, tester_sites), 1)
    result = evaluate_process(
        pt,
        seq;
        tester=memory,
        tester_seq=controls,
        progress=false,
    )
    @test _tester_result_dense(result) ≈ Matrix{ComplexF64}(I, 2, 2) / 2 atol=1e-9
end

@testset "evaluate_process: mixed controls and boundary ordering" begin
    system_sites = siteinds("Qubit", 1)
    tester_sites, memory = _qubit_tester()
    dt = 0.12
    pt = build_process_tensor(
        qubit_system(system_sites);
        dt,
        nsteps=4,
        progress=false,
    )
    rho0 = to_dm(MPS(system_sites, ["0"]))
    closed = _tester_schedule(pt, rho0; open=false)

    mixed = TesterSeq(nsteps=4)
    add!(mixed, tester_propagation(OpSum() + (0.2, "X", 1), tester_sites), 1)
    add!(mixed, tester_unitary(op("H", only(tester_sites)), tester_sites), 2)
    add!(
        mixed,
        joint_propagation(
            OpSum() + (0.3, "Z", 1, "Z", 2),
            system_sites,
            tester_sites,
        ),
        3,
    )
    add!(
        mixed,
        joint_unitary(
            op("CNOT", only(system_sites), only(tester_sites)),
            system_sites,
            tester_sites,
        ),
        4,
    )
    value = evaluate_process(
        pt,
        closed;
        tester=memory,
        tester_seq=mixed,
        progress=false,
    )
    @test value ≈ 1.0 + 0.0im atol=1e-9

    ordering_pt = build_process_tensor(
        qubit_system(system_sites);
        dt,
        nsteps=2,
        progress=false,
    )
    rho_ordering = to_dm(MPS(system_sites, ["+"]))
    ordering_seq = _tester_schedule(ordering_pt, rho_ordering; open=true)
    H_control = OpSum() + (0.9, "Z", 1)
    add!(ordering_seq, unitary_propagation(H_control, ordering_pt.system.sites), 1)
    H_joint = OpSum() + (0.7, "X", 1, "X", 2)
    ordering_controls = TesterSeq(nsteps=2)
    add!(
        ordering_controls,
        joint_propagation(H_joint, system_sites, tester_sites),
        1,
    )
    ordered_result = _tester_result_dense(evaluate_process(
        ordering_pt,
        ordering_seq;
        tester=memory,
        tester_seq=ordering_controls,
        progress=false,
    ))

    joint_sites = vcat(system_sites, tester_sites)
    U_joint = _exact_unitary_exp(H_joint, joint_sites, dt)
    U_system = _exact_unitary_exp(H_control, system_sites, dt)
    U_control = kron(Matrix{ComplexF64}(I, 2, 2), U_system)
    rho_joint0 = kron(
        ComplexF64[1 0; 0 0],
        ComplexF64[0.5 0.5; 0.5 0.5],
    )
    ordered_density = U_control * U_joint * rho_joint0 * U_joint' * U_control'
    swapped_density = U_joint * U_control * rho_joint0 * U_control' * U_joint'
    ordered_reference = _partial_trace_env(ordered_density, 2, 2)
    swapped_reference = _partial_trace_env(swapped_density, 2, 2)
    @test ordered_result ≈ ordered_reference atol=1e-9
    @test !isapprox(ordered_reference, swapped_reference; atol=1e-5)
end

@testset "evaluate_process: tester validation" begin
    system_sites = siteinds("Qubit", 1)
    tester_sites, memory = _qubit_tester()
    pt = build_process_tensor(
        qubit_system(system_sites);
        dt=0.1,
        nsteps=2,
        progress=false,
    )
    rho0 = to_dm(MPS(system_sites, ["0"]))
    seq = _tester_schedule(pt, rho0; open=false)

    @test_throws ArgumentError evaluate_process(
        pt,
        seq;
        tester_seq=TesterSeq(nsteps=pt.nsteps),
        progress=false,
    )
    @test_throws ArgumentError evaluate_process(
        pt,
        seq;
        tester=memory,
        tester_seq=TesterSeq(nsteps=pt.nsteps + 1),
        progress=false,
    )

    initial_joint = TesterSeq(nsteps=pt.nsteps)
    add!(
        initial_joint,
        joint_propagation(
            OpSum() + (0.1, "X", 1, "X", 2),
            system_sites,
            tester_sites,
        ),
        0,
    )
    @test_throws ArgumentError evaluate_process(
        pt,
        seq;
        tester=memory,
        tester_seq=initial_joint,
        progress=false,
    )

    nonunitary = 2 * op("X", only(tester_sites))
    invalid = TesterSeq(nsteps=pt.nsteps)
    add!(invalid, tester_unitary(nonunitary, tester_sites), 1)
    @test_throws ArgumentError evaluate_process(
        pt,
        seq;
        tester=memory,
        tester_seq=invalid,
        progress=false,
    )
end

@testset "evaluate_process: joint symmetric-splitting convergence" begin
    total_time = 0.6
    omega = 0.8
    coupling = 0.65
    system_sites = siteinds("Qubit", 1)
    tester_sites, memory = _qubit_tester()
    rho0 = to_dm(MPS(system_sites, ["0"]))
    H_system = OpSum() + (omega, "Z", 1)
    H_joint = OpSum() + (coupling, "X", 1, "X", 2)

    function split_result(dt)
        nsteps = round(Int, total_time / dt)
        pt = build_process_tensor(
            qubit_system(system_sites, H_system);
            dt,
            nsteps,
            progress=false,
        )
        seq = _tester_schedule(pt, rho0; open=true)
        controls = TesterSeq(nsteps=nsteps)
        for k in 1:nsteps
            add!(
                controls,
                joint_propagation(H_joint, system_sites, tester_sites),
                k,
            )
        end
        return _tester_result_dense(evaluate_process(
            pt,
            seq;
            tester=memory,
            tester_seq=controls,
            progress=false,
        ))
    end

    full_generator = OpSum()
    full_generator += omega, "Z", 1
    full_generator += coupling, "X", 1, "X", 2
    U_exact = _exact_unitary_exp(
        full_generator,
        vcat(system_sites, tester_sites),
        total_time,
    )
    rho_joint0 = kron(ComplexF64[1 0; 0 0], ComplexF64[1 0; 0 0])
    exact = _partial_trace_env(
        U_exact * rho_joint0 * U_exact',
        2,
        2,
    )
    coarse_error = norm(split_result(0.2) - exact)
    fine_error = norm(split_result(0.1) - exact)
    @test fine_error < 0.4 * coarse_error
end

@testset "evaluate_process: one-mode spin bath vs three-body ED" begin
    function _partial_trace_bath_and_tester(rho::AbstractMatrix)
        r = reshape(ComplexF64.(rho), 2, 2, 2, 2, 2, 2)
        out = zeros(ComplexF64, 2, 2)
        for bath in 1:2, tester in 1:2
            out .+= @view r[:, bath, tester, :, bath, tester]
        end
        return out
    end

    dt = 0.05
    nsteps = 4
    total_time = nsteps * dt
    system_sites = siteinds("Qubit", 1)
    bath_phys = siteinds("S=1/2", 1)
    tester_sites, memory = _qubit_tester()
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
    bath = spin_bath([mode])
    pt = build_process_tensor(
        system;
        environment=bath,
        dt,
        nsteps,
        sys_alg=Trotter{2}(),
        progress=false,
    )

    seq = _tester_schedule(pt, rho0; open=true)
    H_joint = OpSum() + (0.2, "X", 1, "X", 2)
    controls = TesterSeq(nsteps=nsteps)
    for k in 1:nsteps
        add!(controls, joint_propagation(H_joint, system_sites, tester_sites), k)
    end
    result = _tester_result_dense(evaluate_process(
        pt,
        seq;
        tester=memory,
        tester_seq=controls,
        progress=false,
    ))

    joint_sites = Index[only(system_sites), only(bath_phys), only(tester_sites)]
    H_full = OpSum()
    H_full += 0.4, "Z", 1
    H_full += 0.3, "Sx", 2
    H_full += 0.25, "Z", 1, "Sz", 2
    H_full += 0.2, "X", 1, "X", 3
    psi0 = MPS(joint_sites, ["0", "Up", "0"])
    rho0_joint = hilbert_mpo_to_dense(to_dm(psi0), joint_sites)
    U = _exact_unitary_exp(H_full, joint_sites, total_time)
    exact = _partial_trace_bath_and_tester(U * rho0_joint * U')
    @test result ≈ exact atol=1e-3
end
