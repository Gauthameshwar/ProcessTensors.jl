# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: test/process_tensor/test_timestep_convention.jl
# Contributor: Gauthameshwar S.
#
# Temporarily tests one-based PT cores, leg attachments, and snapshot times against
# exact references; these suites will be folded into their owning test files.
#
# Run with:
# julia --project=. test/process_tensor/test_timestep_convention.jl

using ProcessTensors
using ProcessTensors.Instruments: instrument_leg_maps
using ITensors
using LinearAlgebra
using Test

if !isdefined(Main, :hilbert_mpo_to_dense)
    include(joinpath(@__DIR__, "..", "time_evolution", "tebd_test_utils.jl"))
end
if !isdefined(Main, :_exact_unitary_exp)
    include(joinpath(@__DIR__, "pt_ed_test_utils.jl"))
end

@testset "timestep convention" begin
    sys_phys = siteinds("S=1/2", 1)
    H_sys = OpSum() + (0.7, "Sx", 1) + (0.3, "Sz", 1)
    system = spin_system(sys_phys, H_sys)
    rho0_h = to_dm(MPS(sys_phys, ["Up"]))
    rho0 = hilbert_mpo_to_dense(rho0_h, sys_phys)
    O_Sz = OpSum() + (1.0, "Sz", 1)
    O_Sx = OpSum() + (1.0, "Sx", 1)
    O_Sy = OpSum() + (1.0, "Sy", 1)
    dt = 0.1

    env_phys = siteinds("S=1/2", 1)
    env_liouv = liouv_sites(env_phys)
    rho_env0_h = to_dm(MPS(env_phys, ["Up"]))
    rho_env = hilbert_mpo_to_dense(rho_env0_h, env_phys)
    mode = spin_mode(
        env_liouv,
        OpSum() + (1.0, "Sx", 1),
        to_liouville(rho_env0_h; sites=env_liouv);
        coupling=OpSum() + (1.0, "Sz", 1, "Sz", 2),
    )
    bath = spin_bath([mode])
    H_bg = OpSum() + (1.0, "Sx", 2) + (1.0, "Sz", 1, "Sz", 2)
    joint_sites = _joint_phys_sites(sys_phys, env_phys)

    @testset "parent test_structure: core bounds and tstep tags" begin
        pt = build_process_tensor(system, only(system.sites); environment=bath, dt, nsteps=3, progress=false)
        @test length(pt.core) == 3
        for k in 1:3
            out_k = only(output_sites(pt, k))
            in_k = only(input_sites(pt, k))
            @test tag_value(out_k, "tstep=") == string(k)
            @test in_k == prime(out_k)
            @test out_k in inds(pt.core[k])
            @test in_k in inds(pt.core[k])
        end
        @test_throws BoundsError output_sites(pt, 0)
        @test_throws BoundsError input_sites(pt, 0)
        @test_throws BoundsError output_sites(pt, 4)
        @test_throws BoundsError input_sites(pt, 4)
        for k in 1:2
            out_prev, in_next = coupling_times(pt, k)
            @test only(out_prev) == only(output_sites(pt, k))
            @test only(in_next) == only(input_sites(pt, k + 1))
        end
        @test_throws BoundsError coupling_times(pt, 0)
        @test_throws BoundsError coupling_times(pt, 3)
    end

    @testset "parent test_instruments: attachments, overlap, and coverage" begin
        pt1 = build_process_tensor(system; dt, nsteps=1, progress=false)
        U = _exact_unitary_exp(H_sys, sys_phys, dt)
        rho_dt = U * rho0 * U'

        seq_prep_first = InstrumentSeq(default=identity_operation(), nsteps=1)
        add!(seq_prep_first, state_preparation(rho0_h), 1)
        add!(seq_prep_first, trace_out(), 1)
        seq_trace_first = InstrumentSeq(default=identity_operation(), nsteps=1)
        add!(seq_trace_first, trace_out(), 1)
        add!(seq_trace_first, state_preparation(rho0_h), 1)
        @test evaluate_process(pt1, seq_prep_first; progress=false) ≈ 1 atol=1e-12
        @test evaluate_process(pt1, seq_trace_first; progress=false) ≈ 1 atol=1e-12

        seq_obs = InstrumentSeq(default=identity_operation(), nsteps=1)
        add!(seq_obs, observable_measurement(O_Sz), 1)
        add!(seq_obs, state_preparation(rho0_h), 1)
        @test evaluate_process(pt1, seq_obs; progress=false) ≈
              tr(dense_hamiltonian_matrix(O_Sz, sys_phys) * rho_dt) atol=1e-12

        seq = InstrumentSeq(default=identity_operation(), nsteps=3)
        @test_throws ArgumentError add!(seq, state_preparation(rho0_h), 0)
        @test_throws ArgumentError add!(seq, trace_out(), 4)
        @test_throws ArgumentError add!(seq, identity_operation(), 3)
        @test_throws ArgumentError add!(seq, identity_operation(), 0)
        add!(seq, trace_out(), 3)
        add!(seq, state_preparation(rho0_h), 1)
        in_map, out_map, missing_in, missing_out = instrument_leg_maps(seq, 3)
        @test sort!(collect(keys(in_map))) == [1, 2, 3]
        @test sort!(collect(keys(out_map))) == [1, 2, 3]
        @test out_map[3] isa TraceOut
        @test isempty(missing_in)
        @test isempty(missing_out)

        in_map1, out_map1, _, missing_out1 = instrument_leg_maps(seq_prep_first, 1)
        @test sort!(collect(keys(in_map1))) == [1]
        @test sort!(collect(keys(out_map1))) == [1]
        @test isempty(missing_out1)

        H_control = OpSum() + (0.4, "Sy", 1)
        between_first = InstrumentSeq(default=identity_operation(), nsteps=3)
        add!(between_first, unitary_propagation(H_control, sys_phys), 1)
        @test_throws ArgumentError add!(between_first, observable_measurement(O_Sz), 1)
        @test_throws ArgumentError add!(between_first, observable_measurement(O_Sz; leg_plev=1), 2)
        @test_throws ArgumentError add!(between_first, observable_measurement(O_Sz; leg_plev=1), 1)
        add!(between_first, state_preparation(rho0_h), 1)
        @test between_first.entries[(1, :input)] isa StatePreparation

        output_first = InstrumentSeq(default=identity_operation(), nsteps=3)
        add!(output_first, observable_measurement(O_Sz), 1)
        @test_throws ArgumentError add!(output_first, unitary_propagation(H_control, sys_phys), 1)
        input_first = InstrumentSeq(default=identity_operation(), nsteps=3)
        add!(input_first, observable_measurement(O_Sz; leg_plev=1), 2)
        @test_throws ArgumentError add!(input_first, unitary_propagation(H_control, sys_phys), 1)

        pt3 = build_process_tensor(system; dt, nsteps=3, progress=false)
        split_pair = InstrumentSeq(default=identity_operation(), nsteps=3)
        add!(split_pair, state_preparation(rho0_h), 1)
        add!(split_pair, observable_measurement(O_Sz), 1)
        add!(split_pair, observable_measurement(O_Sx; leg_plev=1), 2)
        add!(split_pair, trace_out(), 3)
        product_pair = InstrumentSeq(default=identity_operation(), nsteps=3)
        add!(product_pair, state_preparation(rho0_h), 1)
        add!(product_pair, observable_measurement(O_Sz) * observable_measurement(O_Sx; leg_plev=1), 1)
        add!(product_pair, trace_out(), 3)
        @test evaluate_process(pt3, split_pair; progress=false) ≈
              evaluate_process(pt3, product_pair; progress=false) atol=1e-12
    end

    @testset "parent test_evolve_contractions: trajectories on 0:dt:nsteps*dt" begin
        U = _exact_unitary_exp(H_sys, sys_phys, dt)
        for nsteps in (1, 3)
            pt = build_process_tensor(system; dt, nsteps, progress=false)
            closures = evolve(pt, rho0_h; progress=false)
            reference = evolve(pt, rho0_h; contraction=:evaluate, progress=false)
            @test closures.times ≈ [k * dt for k in 0:nsteps]
            @test reference.times ≈ closures.times
            @test length(closures.states_hilbert) == nsteps + 1
            @test length(reference.states_hilbert) == nsteps + 1
            for k in 0:nsteps
                exact = U^k * rho0 * (U^k)'
                @test _one_site_hilbert_mpo_to_dense(closures.states_hilbert[k + 1]) ≈ exact atol=1e-10
                @test _one_site_hilbert_mpo_to_dense(reference.states_hilbert[k + 1]) ≈ exact atol=1e-10
            end
        end

        pt = build_process_tensor(system, only(system.sites); environment=bath, dt, nsteps=3, progress=false)
        closures = evolve(pt, rho0_h; progress=false)
        reference = evolve(pt, rho0_h; contraction=:evaluate, progress=false)
        rho_joint0 = kron(ComplexF64.(rho_env), ComplexF64.(rho0))
        for k in 0:3
            rho_k = _evolve_joint_split_exact(rho_joint0, dt, k, H_sys, H_bg, sys_phys, joint_sites; denv=2)
            exact = _partial_trace_env(rho_k, 2, 2)
            @test _one_site_hilbert_mpo_to_dense(closures.states_hilbert[k + 1]) ≈ exact atol=1e-9
            @test _one_site_hilbert_mpo_to_dense(reference.states_hilbert[k + 1]) ≈ exact atol=1e-9
        end
        @test _one_site_hilbert_mpo_to_dense(closures.states_hilbert[1]) ≈ rho0 atol=1e-12
    end

    @testset "parent test_evolve_testers: tester initialization snapshot" begin
        qubit_sites = siteinds("Qubit", 1)
        tester_sites = siteinds("Qubit", 1)
        pt = build_process_tensor(qubit_system(qubit_sites); dt, nsteps=2, progress=false)
        rho_qubit = to_dm(MPS(qubit_sites, ["0"]))
        memory = tester(tester_sites, to_dm(MPS(tester_sites, ["1"])))
        controls = TesterSeq(
            nsteps=2,
            initialization=tester_unitary(op("X", only(tester_sites)), tester_sites),
        )
        @test_throws ArgumentError add!(controls, tester_identity(), 0)
        trajectory = evolve(pt, rho_qubit; tester=memory, tester_seq=controls, return_joint=true, progress=false)
        @test trajectory.times ≈ [0.0, dt, 2dt]
        @test length(trajectory.tester_states_hilbert) == 3
        @test length(trajectory.joint_states_hilbert) == 3
        @test _one_site_hilbert_mpo_to_dense(trajectory.states_hilbert[1]) ≈ ComplexF64[1 0; 0 0] atol=1e-12
        @test _one_site_hilbert_mpo_to_dense(trajectory.tester_states_hilbert[1]) ≈ ComplexF64[1 0; 0 0] atol=1e-12
        for k in 1:3
            @test _one_site_hilbert_mpo_to_dense(trajectory.tester_states_hilbert[k]) ≈ ComplexF64[1 0; 0 0] atol=1e-12
        end
    end

    @testset "parent test_multitime_correlation: snapshot indices" begin
        nsteps = 4
        pt = build_process_tensor(system, only(system.sites); environment=bath, dt, nsteps, progress=false)
        A = dense_hamiltonian_matrix(O_Sz, sys_phys)
        B = dense_hamiltonian_matrix(O_Sx, sys_phys)
        rho_joint0 = kron(ComplexF64.(rho_env), ComplexF64.(rho0))
        corr(op_a, op_b) = evaluate_process(
            pt,
            two_time_correlation_seq(pt, op_a, op_b; rho0=rho0_h);
            progress=false,
        )

        @test corr((O_Sz, 1), (O_Sx, 1)) ≈ tr(A * B * rho0) atol=1e-10

        rho_2dt = _partial_trace_env(
            _evolve_joint_split_exact(rho_joint0, dt, 2, H_sys, H_bg, sys_phys, joint_sites; denv=2),
            2,
            2,
        )
        @test corr((O_Sz, 3), (O_Sx, 3)) ≈ tr(A * B * rho_2dt) atol=1e-10

        rho_4dt = _partial_trace_env(
            _evolve_joint_split_exact(rho_joint0, dt, 4, H_sys, H_bg, sys_phys, joint_sites; denv=2),
            2,
            2,
        )
        @test corr((O_Sz, 5), (O_Sx, 5)) ≈ tr(A * B * rho_4dt) atol=1e-10

        B_then = kron(ComplexF64.(rho_env), B * ComplexF64.(rho0))
        final_vs_initial = _partial_trace_env(
            _evolve_joint_split_exact(B_then, dt, 4, H_sys, H_bg, sys_phys, joint_sites; denv=2),
            2,
            2,
        )
        @test corr((O_Sz, 5), (O_Sx, 1)) ≈ tr(A * final_vs_initial) atol=1e-10

        after_dt = _evolve_joint_split_exact(rho_joint0, dt, 1, H_sys, H_bg, sys_phys, joint_sites; denv=2)
        left = _evolve_joint_split_continue(
            _apply_sys_op_joint(after_dt, O_Sx, sys_phys, 2; side=:left),
            dt, 2, H_sys, H_bg, sys_phys, joint_sites; denv=2,
        )
        @test corr((O_Sz, 4), (O_Sx, 2)) ≈ tr(A * _partial_trace_env(left, 2, 2)) atol=1e-10

        right = _evolve_joint_split_continue(
            _apply_sys_op_joint(after_dt, O_Sy, sys_phys, 2; side=:right),
            dt, 2, H_sys, H_bg, sys_phys, joint_sites; denv=2,
        )
        @test corr((O_Sy, 2), (O_Sz, 4)) ≈ tr(A * _partial_trace_env(right, 2, 2)) atol=1e-10
        @test !isapprox(corr((O_Sy, 2), (O_Sz, 4)), corr((O_Sz, 4), (O_Sy, 2)); atol=1e-6)

        @test_throws ArgumentError two_time_correlation_seq(pt, (O_Sz, 0), (O_Sx, 1); rho0=rho0_h)
        @test_throws ArgumentError two_time_correlation_seq(pt, (O_Sz, nsteps + 2), (O_Sx, 1); rho0=rho0_h)
    end
end
