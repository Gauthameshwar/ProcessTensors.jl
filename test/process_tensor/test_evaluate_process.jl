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
        @test isapprox(val, 1.0; atol=1e-6)

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

        rho_out = evaluate_process(pt, seq)
        @test rho_out isa MPO{Liouville}
        @test length(rho_out.core) == 1

        trj = evolve(pt, rho0_h)
        ρ_ref = _mpo_to_dense(to_hilbert(rho_out))
        ρ_final = _mpo_to_dense(trj.states_hilbert[end])
        @test ρ_ref ≈ ρ_final atol=1e-10
    end

    @testset "OpenOutput intermediate marginal vs evolve" begin
        s = siteinds("S=1/2", 1)
        H = OpSum()
        H += 0.5, "Sz", 1
        system = spin_system(s, H)
        pt = build_process_tensor(
            system;
            dt=0.05,
            nsteps=4,
            embed_system_propagation=false,
        )
        rho0_h = to_dm(MPS(s, ["Up"]))
        sysprop = SystemPropagation(system)

        seq = InstrumentSeq(default=sysprop, nsteps=pt.nsteps)
        add!(seq, StatePreparation(rho0_h), 0)
        add!(seq, OpenOutput(), 2)
        add!(seq, IdentityOperation(), 3)
        add!(seq, TraceOut(), pt.nsteps)

        rho_out = evaluate_process(pt, seq; default_instr=sysprop)
        @test rho_out isa MPO{Liouville}
        @test !all_pt_legs_contracted(pt, seq)

        trj = evolve(pt, rho0_h; default_instr=sysprop)
        ρ_ref = _mpo_to_dense(trj.states_hilbert[2])
        ρ_open = _mpo_to_dense(to_hilbert(rho_out)) / tr(_mpo_to_dense(to_hilbert(rho_out)))
        @test ρ_ref ≈ ρ_open atol=1e-10
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
end
