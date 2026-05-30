# API and smoke tests for single-site process tensors (build, legs, instruments, Markovian path).
using ProcessTensors
using ITensors
using Test

struct _UnsupportedPTInstrument <: AbstractInstrument end

if !isdefined(Main, :liouville_state_to_dense)
    include(joinpath(@__DIR__, "..", "time_evolution", "tebd_test_utils.jl"))
end

function _one_site_hilbert_mpo_to_dense(mpo::AbstractMPO{Hilbert})
    site = only(filter(i -> plev(i) == 0, collect(inds(mpo.core[1]))))
    d = dim(site)
    return reshape(ComplexF64.(Array(mpo.core[1], prime(site), site)), d, d)
end

_one_site_liouville_state_to_dense(ρ::AbstractMPS{Liouville}) =
    _one_site_hilbert_mpo_to_dense(to_hilbert(ρ))

@testset "process_tensor.jl: single-site rebuild API" begin
    @testset "build_process_tensor uses explicit coupling_site::Index" begin
        s = siteinds("S=1/2", 2)
        H = OpSum()
        H += 0.3, "Sz", 1
        H += -0.1, "Sz", 2
        system = spin_system(s, H)

        pt = build_process_tensor(system, system.sites[2]; dt=0.1, nsteps=3)
        @test pt isa ProcessTensor
        @test pt.coupling_site == system.sites[2]
        @test length(pt.core) == pt.nsteps == 3
        @test length(output_sites(pt, 0)) == 1
        @test length(input_sites(pt, 2)) == 1

        @test_throws ArgumentError build_process_tensor(system, Index(dim(system.sites[1])); dt=0.1, nsteps=3)
    end

    @testset "coupling_times / coupling_sites resolve stable PT legs" begin
        s = siteinds("S=1/2", 1)
        system = @test_warn r"SpinSystem: H is empty" spin_system(s, OpSum())
        pt = build_process_tensor(system, system.sites[1]; dt=0.2, nsteps=4)
        out1, in2 = coupling_times(pt, 2)
        @test length(in2) == 1
        @test length(out1) == 1
        @test plev(only(in2)) == 1
        @test plev(only(out1)) == 0
        @test tag_value(only(in2), "tstep=") == "2"
        @test tag_value(only(out1), "tstep=") == "1"

        # Stable object identity across repeated calls (no fresh-mint index drift).
        out1_again, in2_again = coupling_times(pt, 2)
        @test only(out1_again) == only(out1)
        @test only(in2_again) == only(in2)

        in2b, out1b = coupling_sites(pt, 2)
        @test only(in2b) == only(in2)
        @test only(out1b) == only(out1)
    end

    @testset "build_process_tensor supports multi-mode SpinBath under dense budget" begin
        s = siteinds("S=1/2", 1)
        e1 = siteinds("S=1/2", 1)
        e2 = siteinds("S=1/2", 1)
        L_sys = liouv_sites(s)
        L1 = liouv_sites(e1)
        L2 = liouv_sites(e2)
        H_sys = OpSum()
        H_sys += 0.2, "Sz", 1
        sys = spin_system(s, H_sys)

        ρ1 = to_liouville(to_dm(MPS(e1, ["Up"])); sites=L1)
        H_env = OpSum()
        H_env += 1.0, "Sx", 1
        cpl1 = OpSum() + (0.05, "Sz", 1, "Sz", 2)
        m1 = SpinMode(L1, H_env, ρ1; coupling=cpl1)
        ρ2 = to_liouville(to_dm(MPS(e2, ["Up"])); sites=L2)
        cpl2 = OpSum() + (0.03, "Sz", 1, "Sz", 2)
        m2 = SpinMode(L2, H_env, ρ2; coupling=cpl2)
        bath = spin_bath([m1, m2])
        pt = build_process_tensor(sys, sys.sites[1]; environment=bath, dt=0.05, nsteps=2)
        @test pt isa ProcessTensor
        @test length(pt.core) == 2
        trj = evolve(pt, to_dm(MPS(s, ["Up"])))
        @test length(trj.states_liouville) == 2
    end

    @testset "build_process_tensor rejects oversized mixed bath with warning+error" begin
        s = siteinds("S=1/2", 1)
        system = @test_warn r"SpinSystem: H is empty" spin_system(s, OpSum())

        b1 = siteinds("Boson", 1; dim=40)
        b2 = siteinds("Boson", 1; dim=40)
        Lb1 = liouv_sites(b1)
        Lb2 = liouv_sites(b2)

        rho1 = to_liouville(to_dm(MPS(b1, ["0"])); sites=Lb1)
        rho2 = to_liouville(to_dm(MPS(b2, ["0"])); sites=Lb2)
        H_b = OpSum()
        H_b += 0.2, "N", 1
        cpl1 = OpSum() + (0.02, "N", 1, "Sz", 2)
        cpl2 = OpSum() + (0.03, "N", 1, "Sz", 2)
        m1 = bosonic_mode(Lb1, H_b, dim(only(Lb1)) - 1, rho1; coupling=cpl1)
        m2 = bosonic_mode(Lb2, H_b, dim(only(Lb2)) - 1, rho2; coupling=cpl2)
        bath = @test_warn r"BosonicBath has bath-only Liouville dimension" bosonic_bath([m1, m2])

        @test_logs (:warn, r"build_process_tensor: joint Liouville vector dimension D=.*exceeds MAX_DENSE_LIOUVILLE_DIM") begin
            @test_throws ArgumentError build_process_tensor(system, system.sites[1]; environment=bath, dt=0.05, nsteps=2)
        end
    end

    @testset "single-site instrument validation" begin
        s = siteinds("S=1/2", 1)
        psi0 = MPS(s, ["Up"])
        bad_input = [prime(s[1]), prime(s[1])]
        bad_output = [s[1], s[1]]
        op_sz = OpSum()
        op_sz += 1.0, "Sz", 1

        @test_throws ArgumentError StatePreparation(psi0, bad_input; leg_plev=1)
        @test_throws ArgumentError ObservableMeasurement(op_sz, bad_output; leg_plev=0)
        @test_throws ArgumentError TraceOut(bad_output; leg_plev=0)
        @test_throws ArgumentError IdentityOperation(bad_input, bad_output)
    end

    @testset "markovian fallback evolve still works without environment" begin
        s = siteinds("S=1/2", 1)
        H = OpSum()
        H += 0.5, "Sz", 1
        system = spin_system(s, H)
        pt = build_process_tensor(system; dt=0.05, nsteps=3)
        psi0 = MPS(s, ["Up"])
        rho0 = to_liouville(to_dm(psi0); sites=system.sites)

        trajectory = evolve(pt, psi0)
        manual = tebd_trajectory(rho0, H, 0.05, 3; jump_ops=[], maxdim=32, cutoff=1e-12, alg=Trotter{2}())

        @test pt.embed_system_propagation
        @test trajectory.times ≈ [0.0, 0.05, 0.1] atol=1e-12
        @test length(trajectory.states_liouville) == 3
        for i in 1:pt.nsteps
            @test _one_site_liouville_state_to_dense(trajectory.states_liouville[i]) ≈
                  _one_site_liouville_state_to_dense(manual[i + 1]) atol=1e-10
        end
    end

    @testset "lazy APIs reject embed_system_propagation=false" begin
        s = siteinds("S=1/2", 1)
        H = OpSum()
        H += 0.5, "Sz", 1
        system = spin_system(s, H)
        psi0 = MPS(s, ["Up"])
        pt = build_process_tensor(system; dt=0.05, nsteps=3, embed_system_propagation=false)
        @test !pt.embed_system_propagation
        @test_logs (:warn, r"requires embed_system_propagation=true") @test_throws ArgumentError evolve(pt, psi0)
    end

    @testset "ProcessTensor direct single-site constructor" begin
        s = siteinds("S=1/2", 1)
        system = spin_system(s, OpSum() + (0.2, "Sz", 1))
        pt_ref = build_process_tensor(system; dt=0.1, nsteps=3)
        pt = ProcessTensor(pt_ref.core, system, nothing, 0.1, 3)
        @test pt.coupling_site == only(system.sites)
        @test pt.dt == 0.1
        @test pt.nsteps == 3
        @test pt.environment === nothing
        @test length(pt.core) == 3

        s2 = siteinds("S=1/2", 2)
        system2 = spin_system(s2, OpSum() + (0.1, "Sz", 1))
        pt_multi = build_process_tensor(system2, system2.sites[1]; dt=0.1, nsteps=2)
        @test_throws ArgumentError ProcessTensor(pt_multi.core, system2, nothing, 0.1, 2)
    end

    @testset "ProcessTensor core property forwarding" begin
        s = siteinds("S=1/2", 1)
        system = spin_system(s, OpSum() + (0.2, "Sz", 1))
        pt = build_process_tensor(system; dt=0.1, nsteps=3)
        core = pt.core

        @test getproperty(pt, :core) === core
        @test getproperty(pt, :nsteps) == 3
        @test getproperty(pt, :dt) == 0.1

        common_props = filter(sym -> hasproperty(core, sym), (:llim, :rlim, :center, :orthocenter))
        @test !isempty(common_props)
        for prop in common_props
            @test getproperty(pt, prop) == getproperty(core, prop)
            val = getproperty(core, prop)
            @test_nowarn setproperty!(pt, prop, val)
            @test getproperty(pt, prop) == val
            @test getproperty(core, prop) == val
        end
    end

    @testset "create_instruments bond dispatch and invalid types" begin
        # Mirrors the step loop in create_instruments (process_tensor.jl:721-742).
        function _create_instruments_step_bond(pt, instr, step)
            out_prev, in_curr = coupling_times(pt, step)
            if instr isa TwoLegInstrument
                return instrument_itensor(instr, in_curr, out_prev, step; dt=pt.dt)
            elseif instr isa SingleLegInstrument
                if instr.leg_plev == 0
                    return instrument_itensor(instr, out_prev, step)
                else
                    return instrument_itensor(instr, in_curr, step)
                end
            else
                throw(ArgumentError("create_instruments: unsupported instrument $(typeof(instr)) at step=$step."))
            end
        end

        s = siteinds("S=1/2", 1)
        system = spin_system(s, OpSum() + (0.3, "Sz", 1))
        pt = build_process_tensor(system; dt=0.05, nsteps=3)
        rho0_h = to_dm(MPS(s, ["Up"]))

        @test _create_instruments_step_bond(pt, TraceOut(; leg_plev=1), 1) isa ITensor
        @test _create_instruments_step_bond(pt, TraceOut(; leg_plev=1), 2) isa ITensor
        out1, _ = coupling_times(pt, 2)
        op_z = OpSum() + (1.0, "Sz", 1)
        @test instrument_itensor(ObservableMeasurement(op_z; leg_plev=0), out1, 1) isa ITensor

        seq = InstrumentSeq(default=IdentityOperation(), nsteps=pt.nsteps)
        add!(seq, StatePreparation(rho0_h), 0)
        instruments = create_instruments(pt, seq)
        @test length(instruments) == pt.nsteps
        @test instruments[1] ≈ instrument_itensor(StatePreparation(rho0_h), input_sites(pt, 0), 0)
        for step in 1:(pt.nsteps - 1)
            expected = _create_instruments_step_bond(pt, IdentityOperation(), step)
            @test instruments[step + 1] ≈ expected
        end

        err = @test_throws ArgumentError _create_instruments_step_bond(pt, _UnsupportedPTInstrument(), 1)
        @test occursin("unsupported instrument", lowercase(string(err.value)))
    end

    @testset "evolve rho0 + seq convenience overload" begin
        s = siteinds("S=1/2", 1)
        H = OpSum() + (0.5, "Sz", 1)
        system = spin_system(s, H)
        pt = build_process_tensor(system; dt=0.05, nsteps=3)
        rho0_h = to_dm(MPS(s, ["Up"]))

        seq = InstrumentSeq(default=IdentityOperation(), nsteps=pt.nsteps)
        trj_conv = evolve(pt, rho0_h, seq)
        seq_with_prep = InstrumentSeq(default=IdentityOperation(), nsteps=pt.nsteps)
        add!(seq_with_prep, StatePreparation(rho0_h), 0)
        trj_ref = evolve(pt, seq_with_prep)
        trj_short = evolve(pt, rho0_h)

        @test trj_conv.times ≈ trj_ref.times
        @test trj_conv.times ≈ trj_short.times
        for i in 1:pt.nsteps
            @test _one_site_liouville_state_to_dense(trj_conv.states_liouville[i]) ≈
                  _one_site_liouville_state_to_dense(trj_ref.states_liouville[i]) atol=1e-10
            @test _one_site_liouville_state_to_dense(trj_conv.states_liouville[i]) ≈
                  _one_site_liouville_state_to_dense(trj_short.states_liouville[i]) atol=1e-10
        end
    end
end

@testset "process_tensor.jl: pretty printing" begin
    s = siteinds("S=1/2", 1)
    H = OpSum() + (0.5, "Sz", 1)
    system = spin_system(s, H)
    pt_markov = build_process_tensor(system; dt=0.1, nsteps=3)

    out = sprint(show, pt_markov)
    @test out == sprint(show, MIME"text/plain"(), pt_markov)
    @test occursin("3-step ProcessTensor{SpinSystem, Nothing}", out)
    @test occursin("dt=0.1", out)
    @test occursin("t_final=0.3", out)
    @test occursin("maxlinkdim=1", out)
    @test occursin("system:      SpinSystem(nsites=1, dissipative=false)", out)
    @test occursin("environment: none", out)
    @test occursin("core:        MPO{Liouville}(length=3", out)

    L_env = liouv_sites(siteinds("S=1/2", 1))
    e = siteinds("S=1/2", 1)
    ρ = to_liouville(to_dm(MPS(e, ["Up"])); sites=L_env)
    H_env = OpSum() + (1.0, "Sx", 1)
    mode = SpinMode(L_env, H_env, ρ; coupling=OpSum() + (0.05, "Sz", 1, "Sz", 2))
    bath = spin_bath([mode])
    pt_bath = build_process_tensor(system, system.sites[1]; environment=bath, dt=0.1, nsteps=3)

    out_bath = sprint(show, pt_bath)
    @test occursin("3-step ProcessTensor{SpinSystem, SpinBath}", out_bath)
    @test occursin("dt=0.1", out_bath)
    @test occursin("t_final=0.3", out_bath)
    @test occursin("maxlinkdim=4", out_bath)
    @test occursin("environment: SpinBath(nmodes=1, D_bath=4, coupling=true)", out_bath)
    @test occursin("core:        MPO{Liouville}(length=3", out_bath)
end
