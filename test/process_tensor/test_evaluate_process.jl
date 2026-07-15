# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: test/process_tensor/test_evaluate_process.jl
# Contributor: Gauthameshwar S.
#
# Tests process-tensor evaluation against closed Markovian reference schedules.
#
# Run with:
#   julia --project=. test/runtests.jl

using ProcessTensors
using ITensors
using Test
using LinearAlgebra

if !isdefined(Main, :liouville_state_to_dense)
    include(joinpath(@__DIR__, "..", "time_evolution", "tebd_test_utils.jl"))
end

if !isdefined(Main, :_mpo_to_dense)
    function _mpo_to_dense(mpo::AbstractMPO{Hilbert})
        sites = [only(filter(i -> plev(i) == 0, inds(mpo.core[j]))) for j in 1:length(mpo.core)]
        @assert length(sites) == 1
        site = sites[1]
        d = dim(site)
        rho_dense = Array(mpo.core[1], prime(site), site)
        return reshape(ComplexF64.(rho_dense), d, d)
    end
end

function _closed_markovian_seq(pt, rho0_h; nsteps=pt.nsteps)
    seq = InstrumentSeq(default=IdentityOperation(), nsteps=nsteps)
    add!(seq, StatePreparation(rho0_h), 0)
    add!(seq, TraceOut(), nsteps)
    return seq
end

@testset "process_tensor.jl: evaluate_process" begin
    @testset "all_pt_legs_contracted" begin
        s = siteinds("S=1/2", 1)
        H = OpSum()
        H += 0.5, "Sz", 1
        system = spin_system(s, H)
        pt = build_process_tensor(system; dt=0.05, nsteps=3)
        rho0_h = to_dm(MPS(s, ["Up"]))

        seq_open = InstrumentSeq(default=IdentityOperation(), nsteps=pt.nsteps)
        add!(seq_open, StatePreparation(rho0_h), 0)
        @test !all_pt_legs_contracted(pt, seq_open)

        seq_closed = _closed_markovian_seq(pt, rho0_h)
        @test all_pt_legs_contracted(pt, seq_closed)
    end

    @testset "Markovian scalar" begin
        s = siteinds("S=1/2", 1)
        H = OpSum()
        H += 0.5, "Sz", 1
        L = OpSum()
        L += 0.1, "S-", 1
        system = spin_system(s, H; jump_ops=[L])
        pt = build_process_tensor(system; dt=0.05, nsteps=3)
        rho0_h = to_dm(MPS(s, ["Up"]))
        seq = _closed_markovian_seq(pt, rho0_h)

        val = evaluate_process(pt, seq)
        @test val isa ComplexF64
        @test isapprox(real(val), 1.0; atol=1e-6)

        val_kw = evaluate_process(pt, seq; all_legs_contracted=true)
        @test val_kw ≈ val
    end

    @testset "open final leg returns MPO{Liouville}" begin
        s = siteinds("S=1/2", 1)
        H = OpSum()
        H += 0.5, "Sz", 1
        system = spin_system(s, H)
        pt = build_process_tensor(system; dt=0.05, nsteps=3)
        rho0_h = to_dm(MPS(s, ["Up"]))

        seq = InstrumentSeq(default=IdentityOperation(), nsteps=pt.nsteps)
        add!(seq, StatePreparation(rho0_h), 0)
        add!(seq, OpenOutput(), pt.nsteps)

        rho_out = evaluate_process(pt, seq)
        @test rho_out isa MPO{Liouville}
        @test length(rho_out.core) == 1

        trj = evolve(pt, rho0_h)
        ρ_ref = _mpo_to_dense(to_hilbert(rho_out))
        ρ_final = _mpo_to_dense(trj.states_hilbert[end])
        @test ρ_ref ≈ ρ_final atol=1e-10

        # Implicit terminal open (default Identity) remains supported.
        seq_implicit = InstrumentSeq(default=IdentityOperation(), nsteps=pt.nsteps)
        add!(seq_implicit, StatePreparation(rho0_h), 0)
        @test _mpo_to_dense(to_hilbert(evaluate_process(pt, seq_implicit))) ≈ ρ_final atol=1e-10
    end

    @testset "OpenOutput lazy → create_instruments → manual contract vs ED" begin
        if !isdefined(Main, :_physical_sites_from_hilbert_mpo)
            include(joinpath(@__DIR__, "pt_ed_test_utils.jl"))
        end

        # --- Markovian: exact dense ED at t = (nsteps - 1) * dt ---
        s = siteinds("S=1/2", 1)
        H = OpSum() + (0.7, "Sx", 1)
        system = spin_system(s, H)
        dt = 0.05
        nsteps = 4
        pt = build_process_tensor(system; dt=dt, nsteps=nsteps)
        rho0_h = to_dm(MPS(s, ["Up"]))

        seq = InstrumentSeq(default=IdentityOperation(), nsteps=nsteps)
        add!(seq, StatePreparation(rho0_h), 0)
        add!(seq, OpenOutput(), nsteps)

        instruments = create_instruments(pt, seq)
        @test length(instruments) == nsteps + 1
        @test length(inds(instruments[end])) == 0
        @test isapprox(scalar(instruments[end]), 1.0; atol=1e-12)
        for step in 1:(nsteps - 1)
            @test length(inds(instruments[step + 1])) == 2
        end

        result = pt.core[1] * instruments[1]
        for step in 1:(nsteps - 1)
            result *= instruments[step + 1]
            result *= pt.core[step + 1]
        end
        result *= instruments[end]
        keep = output_sites(pt, nsteps - 1)
        @test length(inds(result)) == 1
        @test only(inds(result)) == only(keep)

        rho_manual_h = to_hilbert(ProcessTensors._liouville_mps_from_itensor(result, keep))
        rho_manual = hilbert_mpo_to_dense(
            rho_manual_h, _physical_sites_from_hilbert_mpo(rho_manual_h),
        )
        rho_eval_h = to_hilbert(evaluate_process(pt, seq))
        rho_eval = hilbert_mpo_to_dense(
            rho_eval_h, _physical_sites_from_hilbert_mpo(rho_eval_h),
        )
        @test rho_manual ≈ rho_eval atol=1e-12

        # Always-embedded PT: each of the `nsteps` cores applies one system map of
        # duration `dt`, so the final open output is the state at physical time
        # `nsteps * dt` (evolve labels that snapshot as `(nsteps - 1) * dt`).
        t_phys = nsteps * dt
        H_mat = dense_hamiltonian_matrix(H, s)
        rho0 = hilbert_mpo_to_dense(rho0_h, s)
        U = exp(-im * t_phys * H_mat)
        rho_ed = U * rho0 * U'
        @test rho_manual ≈ rho_ed atol=1e-10
        @test rho_eval ≈ rho_ed atol=1e-10

        trj = evolve(pt, rho0_h)
        rho_evolve = hilbert_mpo_to_dense(
            trj.states_hilbert[end],
            _physical_sites_from_hilbert_mpo(trj.states_hilbert[end]),
        )
        @test rho_manual ≈ rho_evolve atol=1e-12

        # --- Bath PT: same OpenOutput path must match evolve (split PT reference) ---
        env_phys = siteinds("S=1/2", 1)
        env_liouv = liouv_sites(env_phys)
        rho_env0_l = to_liouville(to_dm(MPS(env_phys, ["Up"])); sites=env_liouv)
        H_env = OpSum() + (0.5, "Sx", 1)
        cpl = OpSum() + (0.3, "Sz", 1, "Sz", 2)
        bath = spin_bath([spin_mode(env_liouv, H_env, rho_env0_l; coupling=cpl)])
        pt_b = build_process_tensor(system, system.sites[1]; environment=bath, dt=dt, nsteps=nsteps)
        seq_b = InstrumentSeq(default=IdentityOperation(), nsteps=nsteps)
        add!(seq_b, StatePreparation(rho0_h), 0)
        add!(seq_b, OpenOutput(), nsteps)
        inst_b = create_instruments(pt_b, seq_b)
        @test length(inds(inst_b[end])) == 0
        @test isapprox(scalar(inst_b[end]), 1.0; atol=1e-12)

        result_b = pt_b.core[1] * inst_b[1]
        for step in 1:(nsteps - 1)
            result_b *= inst_b[step + 1]
            result_b *= pt_b.core[step + 1]
        end
        result_b *= inst_b[end]
        keep_b = output_sites(pt_b, nsteps - 1)
        @test length(inds(result_b)) == 1
        @test only(inds(result_b)) == only(keep_b)
        rho_b_manual_h = to_hilbert(ProcessTensors._liouville_mps_from_itensor(result_b, keep_b))
        rho_b_manual = hilbert_mpo_to_dense(
            rho_b_manual_h, _physical_sites_from_hilbert_mpo(rho_b_manual_h),
        )
        rho_b_eval_h = to_hilbert(evaluate_process(pt_b, seq_b))
        rho_b_eval = hilbert_mpo_to_dense(
            rho_b_eval_h, _physical_sites_from_hilbert_mpo(rho_b_eval_h),
        )
        trj_b = evolve(pt_b, rho0_h)
        rho_b_evolve = hilbert_mpo_to_dense(
            trj_b.states_hilbert[end],
            _physical_sites_from_hilbert_mpo(trj_b.states_hilbert[end]),
        )
        @test rho_b_manual ≈ rho_b_eval atol=1e-12
        @test rho_b_manual ≈ rho_b_evolve atol=1e-12
    end

    @testset "bath PT scalar" begin
        s = siteinds("S=1/2", 1)
        e = siteinds("S=1/2", 1)
        L_sys = liouv_sites(s)
        L_env = liouv_sites(e)
        system = spin_system(s, OpSum() + (0.3, "Sz", 1))
        ρ_env = to_liouville(to_dm(MPS(e, ["Up"])); sites=L_env)
        H_env = OpSum() + (0.5, "Sx", 1)
        cpl = OpSum() + (0.1, "Sz", 1, "Sz", 2)
        mode = SpinMode(L_env, H_env, ρ_env; coupling=cpl)
        bath = spin_bath([mode])
        pt = build_process_tensor(system, system.sites[1]; environment=bath, dt=0.05, nsteps=2)
        rho0_h = to_dm(MPS(s, ["Up"]))
        seq = _closed_markovian_seq(pt, rho0_h; nsteps=pt.nsteps)

        val = evaluate_process(pt, seq)
        @test val isa ComplexF64
        @test isfinite(val)
        @test isapprox(real(val), 1.0; atol=1e-9)
    end

    @testset "observable scalar vs evolve" begin
        s = siteinds("S=1/2", 1)
        system = spin_system(s, OpSum() + (0.3, "Sz", 1))
        pt = build_process_tensor(system; dt=0.05, nsteps=3)
        rho0_h = to_dm(MPS(s, ["Up"]))

        O = OpSum() + (1.0, "Sz", 1)
        seq = InstrumentSeq(default=IdentityOperation(), nsteps=pt.nsteps)
        add!(seq, StatePreparation(rho0_h), 0)
        add!(seq, ObservableMeasurement(O), pt.nsteps)

        val = evaluate_process(pt, seq)
        @test val isa ComplexF64
        @test isapprox(val, 0.5; atol=1e-9)
    end

    @testset "leg-count mismatch error" begin
        s = siteinds("S=1/2", 1)
        system = spin_system(s, OpSum() + (0.2, "Sz", 1))
        pt = build_process_tensor(system; dt=0.05, nsteps=2)
        rho0_h = to_dm(MPS(s, ["Up"]))
        seq = InstrumentSeq(default=IdentityOperation(), nsteps=pt.nsteps)
        add!(seq, StatePreparation(rho0_h), 0)

        err = @test_throws ArgumentError evaluate_process(pt, seq; all_legs_contracted=true)
        @test occursin("expected 0", string(err.value))
        @test occursin("found 1", string(err.value))
    end

    @testset "batch scalar schedules" begin
        s = siteinds("S=1/2", 1)
        system = spin_system(s, OpSum() + (0.4, "Sz", 1))
        pt = build_process_tensor(system; dt=0.05, nsteps=3)
        rho0_h = to_dm(MPS(s, ["Up"]))

        seq1 = _closed_markovian_seq(pt, rho0_h; nsteps=pt.nsteps)
        O = OpSum() + (1.0, "Sx", 1)
        seq2 = InstrumentSeq(default=IdentityOperation(), nsteps=pt.nsteps)
        add!(seq2, StatePreparation(rho0_h), 0)
        add!(seq2, ObservableMeasurement(O), pt.nsteps)

        batch = evaluate_process(pt, [seq1, seq2])
        @test batch isa Vector{ComplexF64}
        @test length(batch) == 2
        @test batch[1] ≈ evaluate_process(pt, seq1)
        @test batch[2] ≈ evaluate_process(pt, seq2)
    end

    @testset "rho0 convenience overloads" begin
        s = siteinds("S=1/2", 1)
        system = spin_system(s, OpSum() + (0.3, "Sz", 1))
        pt = build_process_tensor(system; dt=0.05, nsteps=3)
        rho0_h = to_dm(MPS(s, ["Up"]))

        seq_closed = _closed_markovian_seq(pt, rho0_h)
        @test evaluate_process(pt, rho0_h, seq_closed) ≈ evaluate_process(pt, seq_closed)

        seq_open = InstrumentSeq(default=IdentityOperation(), nsteps=pt.nsteps)
        rho_out = evaluate_process(pt, rho0_h, seq_open)
        seq_manual = InstrumentSeq(default=IdentityOperation(), nsteps=pt.nsteps)
        add!(seq_manual, StatePreparation(rho0_h), 0)
        @test rho_out ≈ evaluate_process(pt, seq_manual)

        rho_default = evaluate_process(pt, rho0_h)
        @test rho_default isa MPO{Liouville}
        trj = evolve(pt, rho0_h)
        ρ_ref = _mpo_to_dense(to_hilbert(rho_default))
        ρ_final = _mpo_to_dense(trj.states_hilbert[end])
        @test ρ_ref ≈ ρ_final atol=1e-10
    end

    @testset "OpenInOut returns multi-leg ITensor" begin
        s = siteinds("S=1/2", 1)
        system = spin_system(s, OpSum() + (0.5, "Sz", 1))
        pt = build_process_tensor(system; dt=0.05, nsteps=3)
        rho0_h = to_dm(MPS(s, ["Up"]))

        seq = InstrumentSeq(default=IdentityOperation(), nsteps=pt.nsteps)
        add!(seq, StatePreparation(rho0_h), 0)
        add!(seq, OpenInOut(), 1)
        add!(seq, TraceOut(), pt.nsteps)

        info = open_leg_info(pt, seq)
        @test info.n_open_expected == 2
        @test 1 in info.open_in
        @test 0 in info.open_out
        @test !all_pt_legs_contracted(pt, seq)

        result = evaluate_process(pt, seq)
        @test result isa ITensor
        @test length(inds(result)) == 2
        @test length(inds(result)) == info.n_open_expected
        @test Set(dim.(inds(result))) == Set(info.open_dims)
    end

    @testset "OpenInput via ProductInstrument leaves one open input" begin
        s = siteinds("S=1/2", 1)
        system = spin_system(s, OpSum() + (0.4, "Sz", 1))
        pt = build_process_tensor(system; dt=0.05, nsteps=3)
        rho0_h = to_dm(MPS(s, ["Up"]))

        seq = InstrumentSeq(default=IdentityOperation(), nsteps=pt.nsteps)
        add!(seq, StatePreparation(rho0_h), 0)
        add!(seq, TraceOut() * OpenInput(), 1)
        add!(seq, TraceOut(), pt.nsteps)

        info = open_leg_info(pt, seq)
        @test info.n_open_expected == 1
        @test 1 in info.open_in
        @test !all_pt_legs_contracted(pt, seq)

        result = evaluate_process(pt, seq)
        @test result isa MPO{Liouville}
        @test length(result.core) == 1
    end

    @testset "verbose summary reports final open-leg counts and shape" begin
        s = siteinds("S=1/2", 1)
        system = spin_system(s, OpSum() + (0.5, "Sz", 1))
        pt = build_process_tensor(system; dt=0.05, nsteps=3)
        rho0_h = to_dm(MPS(s, ["Up"]))

        seq_scalar = InstrumentSeq(default=IdentityOperation(), nsteps=pt.nsteps)
        add!(seq_scalar, StatePreparation(rho0_h), 0)
        add!(seq_scalar, ObservableMeasurement(OpSum() + (1.0, "Sz", 1), output_sites(pt, pt.nsteps - 1)), pt.nsteps)
        @test_logs (
            :info,
            r"Evaluated process",
        ) match_mode = :any evaluate_process(
            pt,
            seq_scalar;
            progress=false,
            verbose=true,
        )

        seq_open = InstrumentSeq(default=IdentityOperation(), nsteps=pt.nsteps)
        add!(seq_open, StatePreparation(rho0_h), 0)
        add!(seq_open, OpenInOut(), 1)
        add!(seq_open, TraceOut(), pt.nsteps)
        result = evaluate_process(pt, seq_open; progress=false, verbose=false)
        idxs = collect(inds(result))
        n_in, n_out = ProcessTensors._open_pt_leg_counts(idxs)
        @test n_in == 1
        @test n_out == 1
        @test ProcessTensors._evaluate_result_shape_text(idxs) == string(Tuple(dim.(idxs)))
    end
end
