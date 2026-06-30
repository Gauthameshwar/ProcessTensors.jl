# 1 system qubit + 1 bath spin: split PT evolution vs exact joint `exp(-im * t * H_full)`.
using ProcessTensors
using ITensors
using Test
using LinearAlgebra

if !isdefined(Main, :liouville_state_to_dense)
    include(joinpath(@__DIR__, "..", "time_evolution", "tebd_test_utils.jl"))
end
if !isdefined(Main, :_physical_sites_from_hilbert_mpo)
    include(joinpath(@__DIR__, "pt_ed_test_utils.jl"))
end

@testset "process tensor: 1+1 spin TFI PT vs joint full-H exact ED" begin
    sys_phys = siteinds("S=1/2", 1)
    env_phys = siteinds("S=1/2", 1)
    sys_liouv = liouv_sites(sys_phys)
    env_liouv = liouv_sites(env_phys)

    H_sys = OpSum()
    H_sys += 1.0, "Sx", 1
    system = spin_system(sys_phys, H_sys)

    rho_env0_h = to_dm(MPS(env_phys, ["Up"]))
    rho_env0_l = to_liouville(rho_env0_h; sites=env_liouv)
    H_env = OpSum()
    H_env += 1.0, "Sx", 1
    cpl = OpSum() + (1.0, "Sz", 1, "Sz", 2)
    mode = spin_mode(env_liouv, H_env, rho_env0_l; coupling=cpl)
    bath = spin_bath([mode])

    dt = 0.1
    nsteps = 8
    pt = build_process_tensor(system, system.sites[1]; environment=bath, dt=dt, nsteps=nsteps)

    rho_sys0_h = to_dm(MPS(sys_phys, ["Up"]))
    trajectory = evolve(pt, rho_sys0_h)

    H_bg = OpSum()
    H_bg += 1.0, "Sx", 2
    H_bg += 1.0, "Sz", 1, "Sz", 2
    joint_sites = _joint_phys_sites(sys_phys, env_phys)
    H_full = _build_joint_full_opsum(H_sys, H_bg)
    rho_joint = kron(
        hilbert_mpo_to_dense(rho_sys0_h, sys_phys),
        hilbert_mpo_to_dense(rho_env0_h, env_phys),
    )

    @test length(trajectory.states_liouville) == nsteps
    split_errs = Float64[]
    joint_errs = Float64[]
    for k in 0:(nsteps - 1)
        t = k * dt
        rho_joint_full = _evolve_joint_full_exact(rho_joint, t, H_full, joint_sites)
        rho_ed = _partial_trace_env(rho_joint_full, 2, 2)

        rho_joint_split = _evolve_joint_split_exact(
            rho_joint, dt, k, H_sys, H_bg, sys_phys, joint_sites; denv=2,
        )
        rho_split = _partial_trace_env(rho_joint_split, 2, 2)
        push!(split_errs, norm(rho_split - rho_ed))

        rho_pt_h = to_hilbert(trajectory.states_liouville[k + 1])
        rho_pt = hilbert_mpo_to_dense(rho_pt_h, _physical_sites_from_hilbert_mpo(rho_pt_h))
        push!(joint_errs, norm(rho_pt - rho_ed))
    end
    # Joint ED is exact; embedded split PT (system + bath every slab) vs full joint at moderate Δt.
    @test maximum(split_errs) < 0.08
    @test maximum(joint_errs) < 0.08

    O_sys = OpSum() + (1.0, "Sz", 1)
    default_instr = _schedule_default_instr_pt(pt)
    obs_errs = Float64[]
    trace_errs = Float64[]
    density_errs = Float64[]

    for k in 0:(nsteps - 1)
        t = k * dt
        rho_ed = _reduced_system_joint_full(rho_joint, t, H_full, joint_sites, 2, 2)

        pt_k = build_process_tensor(system, system.sites[1]; environment=bath, dt=dt, nsteps=k + 1)
        seq_obs = _seq_observable_terminal(rho_sys0_h, O_sys, k + 1, default_instr)
        val_obs = evaluate_process(pt_k, seq_obs; default_instr=default_instr)
        @test val_obs isa ComplexF64
        push!(obs_errs, abs(val_obs - _ed_expectation(rho_ed, O_sys, sys_phys)))

        seq_tr = _seq_trace_terminal(rho_sys0_h, k + 1, default_instr)
        val_tr = evaluate_process(pt_k, seq_tr; default_instr=default_instr)
        @test val_tr isa ComplexF64
        push!(trace_errs, abs(val_tr - real(tr(rho_ed))))

        seq_rho = InstrumentSeq(default=default_instr, nsteps=k + 1)
        add!(seq_rho, StatePreparation(rho_sys0_h), 0)
        rho_pt_h = to_hilbert(evaluate_process(pt_k, seq_rho; default_instr=default_instr))
        rho_pt = _hilbert_mpo_to_dense_one_site(rho_pt_h)
        push!(density_errs, norm(rho_pt - rho_ed))

        rho_ev = _hilbert_mpo_to_dense_one_site(trajectory.states_hilbert[k + 1])
        val_evolve = _ed_expectation(rho_ev, O_sys, sys_phys)
        @test isapprox(val_obs, val_evolve; atol=1e-9, rtol=1e-7)
    end

    @test maximum(obs_errs) < 0.08
    @test maximum(trace_errs) < 0.08
    @test maximum(density_errs) < 0.08
end

@testset "process tensor: zero coupling matches system-only TEBD" begin
    sys_phys = siteinds("S=1/2", 1)
    env_phys = siteinds("S=1/2", 1)
    env_liouv = liouv_sites(env_phys)

    H_sys = OpSum() + (0.65, "Sx", 1)
    system = spin_system(sys_phys, H_sys)

    rho_env_l = to_liouville(to_dm(MPS(env_phys, ["Up"])); sites=env_liouv)
    H_env = OpSum() + (0.9, "Sx", 1)
    mode = spin_mode(env_liouv, H_env, rho_env_l; coupling=OpSum())
    bath = @test_warn r"SpinBath: no mode-system coupling" spin_bath([mode])

    dt = 0.05
    nsteps = 4
    pt = build_process_tensor(system, system.sites[1]; environment=bath, dt=dt, nsteps=nsteps)
    rho0_h = to_dm(MPS(sys_phys, ["Up"]))
    rho0_l = to_liouville(rho0_h; sites=system.sites)

    trj_pt = evolve(pt, rho0_h)
    trj_sys = tebd_trajectory(
        rho0_l,
        H_sys,
        dt,
        nsteps;
        jump_ops=[],
        maxdim=32,
        cutoff=1e-12,
        alg=Trotter{2}(),
    )

    for i in 1:nsteps
        ρ_pt = _one_site_liouville_state_to_dense(trj_pt.states_liouville[i])
        ρ_ref = liouville_state_to_dense(trj_sys[i + 1], sys_phys)
        @test ρ_pt ≈ ρ_ref atol=1e-9 rtol=1e-8
    end
end

@testset "process tensor: empty H_sys coupled bath matches joint ED (|Up> initial)" begin
    sys_phys = siteinds("S=1/2", 1)
    env_phys = siteinds("S=1/2", 1)
    env_liouv = liouv_sites(env_phys)

    system = @test_warn r"SpinSystem: H is empty" spin_system(sys_phys, OpSum())
    rho_env0_h = to_dm(MPS(env_phys, ["Up"]))
    rho_env0_l = to_liouville(rho_env0_h; sites=env_liouv)
    H_env = OpSum() + (1.0, "Sx", 1)
    cpl = OpSum() + (1.0, "Sz", 1, "Sz", 2)
    mode = spin_mode(env_liouv, H_env, rho_env0_l; coupling=cpl)
    bath = spin_bath([mode])

    dt = 0.1
    nsteps = 8
    pt = build_process_tensor(system, system.sites[1]; environment=bath, dt=dt, nsteps=nsteps)
    rho_sys0_h = to_dm(MPS(sys_phys, ["Up"]))
    trajectory = evolve(pt, rho_sys0_h)

    H_bg = OpSum() + (1.0, "Sx", 2) + (1.0, "Sz", 1, "Sz", 2)
    joint_sites = _joint_phys_sites(sys_phys, env_phys)
    rho_joint = kron(
        hilbert_mpo_to_dense(rho_sys0_h, sys_phys),
        hilbert_mpo_to_dense(rho_env0_h, env_phys),
    )

    for k in 0:(nsteps - 1)
        t = k * dt
        rho_ed = _reduced_system_joint_full(rho_joint, t, H_bg, joint_sites, 2, 2)
        rho_pt_h = to_hilbert(trajectory.states_liouville[k + 1])
        rho_pt = hilbert_mpo_to_dense(rho_pt_h, _physical_sites_from_hilbert_mpo(rho_pt_h))
        @test rho_pt ≈ rho_ed atol=1e-10 rtol=1e-9
    end
end

@testset "process tensor: fixed t_final split-schedule error decreases with Δt" begin
    sys_phys = siteinds("S=1/2", 1)
    env_phys = siteinds("S=1/2", 1)
    env_liouv = liouv_sites(env_phys)

    H_sys = OpSum() + (1.0, "Sx", 1)
    system = spin_system(sys_phys, H_sys)
    rho_env0_h = to_dm(MPS(env_phys, ["Up"]))
    rho_env0_l = to_liouville(rho_env0_h; sites=env_liouv)
    H_env = OpSum() + (1.0, "Sx", 1)
    cpl = OpSum() + (1.0, "Sz", 1, "Sz", 2)
    mode = spin_mode(env_liouv, H_env, rho_env0_l; coupling=cpl)
    bath = spin_bath([mode])

    rho_sys0_h = to_dm(MPS(sys_phys, ["Up"]))
    H_bg = OpSum() + (1.0, "Sx", 2) + (1.0, "Sz", 1, "Sz", 2)
    joint_sites = _joint_phys_sites(sys_phys, env_phys)
    H_full = _build_joint_full_opsum(H_sys, H_bg)
    rho_joint = kron(
        hilbert_mpo_to_dense(rho_sys0_h, sys_phys),
        hilbert_mpo_to_dense(rho_env0_h, env_phys),
    )

    # Same physical snapshot time t_final = (n - 1) * dt = 0.4.
    grids = ((5, 0.1), (9, 0.05))
    t_final = 0.4
    errs = Float64[]
    for (nsteps, dt) in grids
        @test isapprox((nsteps - 1) * dt, t_final; atol=1e-12)
        pt = build_process_tensor(
            system, system.sites[1]; environment=bath, dt=dt, nsteps=nsteps,
        )
        trj = evolve(pt, rho_sys0_h)
        rho_ed = _reduced_system_joint_full(rho_joint, t_final, H_full, joint_sites, 2, 2)
        rho_pt_h = to_hilbert(trj.states_liouville[end])
        rho_pt = hilbert_mpo_to_dense(rho_pt_h, _physical_sites_from_hilbert_mpo(rho_pt_h))
        push!(errs, norm(rho_pt - rho_ed))
    end
    # Lie–Trotter split between system and bath steps: first order in Δt at fixed t_final.
    @test errs[2] < errs[1]
    @test errs[2] < 0.08
end

@testset "process tensor: empty H_sys diagonal Sz coupling leaves |Up> invariant" begin
    sys_phys = siteinds("S=1/2", 1)
    env_phys = siteinds("S=1/2", 1)
    env_liouv = liouv_sites(env_phys)

    system = @test_warn r"SpinSystem: H is empty" spin_system(sys_phys, OpSum())
    rho_env_l = to_liouville(to_dm(MPS(env_phys, ["Up"])); sites=env_liouv)
    cpl = OpSum() + (0.25, "Sz", 1, "Sz", 2)
    mode = @test_warn r"SpinMode:H is empty" spin_mode(
        env_liouv, OpSum(), rho_env_l; coupling=cpl,
    )
    bath = spin_bath([mode])

    pt = build_process_tensor(system, system.sites[1]; environment=bath, dt=0.05, nsteps=4)
    rho_up_h = to_dm(MPS(sys_phys, ["Up"]))
    rho_up = hilbert_mpo_to_dense(rho_up_h, sys_phys)

    trj = evolve(pt, rho_up_h)
    for rho_l in trj.states_liouville
        rho_h = to_hilbert(rho_l)
        rho_pt = hilbert_mpo_to_dense(rho_h, _physical_sites_from_hilbert_mpo(rho_h))
        @test rho_pt ≈ rho_up atol=1e-6 rtol=1e-6
    end
end
