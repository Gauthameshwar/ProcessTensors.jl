# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: test/process_tensor/test_ed_tfi_vs_dense.jl
# Contributor: Gauthameshwar S.
#
# Tests single-mode spin-bath process-tensor evolution against joint dense ED.
#
# Run with:
#   julia --project=. test/runtests.jl

using ProcessTensors
using ITensors
using ITensors.Ops: Exact, Trotter
using Test
using LinearAlgebra

if !isdefined(Main, :liouville_state_to_dense)
    include(joinpath(@__DIR__, "..", "time_evolution", "tebd_test_utils.jl"))
end
if !isdefined(Main, :_physical_sites_from_hilbert_mpo)
    include(joinpath(@__DIR__, "pt_ed_test_utils.jl"))
end

@testset "process tensor: bath chronology matches repeated exact channel" begin
    sys_phys = siteinds("S=1/2", 1)
    env_phys = siteinds("S=1/2", 1)
    env_liouv = liouv_sites(env_phys)
    system = @test_logs (:warn, r"SpinSystem: H is empty") spin_system(sys_phys, OpSum())

    ψ_sys = ComplexF64[1, im] / sqrt(2)
    ψ_env = ComplexF64[1, 1] / sqrt(2)
    ρ_sys = ψ_sys * ψ_sys'
    ρ_env = ψ_env * ψ_env'
    H_bg = OpSum()
    for op in ("Sx", "Sy", "Sz")
        H_bg += 1.0, op, 2, op, 1
    end
    coupling = OpSum()
    for op in ("Sx", "Sy", "Sz")
        coupling += 1.0, op, 1, op, 2
    end
    mode = @test_logs (:warn, r"SpinMode:H is empty") spin_mode(
        env_liouv,
        OpSum(),
        to_liouville(hilbert_matrix_to_mpo(ρ_env, env_phys); sites=env_liouv);
        coupling,
    )

    dt, nsteps = 0.1, 6
    pt = build_process_tensor(
        system;
        method=Dense(),
        environment=spin_bath([mode]),
        dt,
        nsteps,
    )
    trajectory = evolve(pt, hilbert_matrix_to_mpo(ρ_sys, sys_phys))
    U_bg = _exact_unitary_exp(H_bg, _joint_phys_sites(sys_phys, env_phys), dt)
    ρ_joint = kron(ρ_env, ρ_sys)
    for k in 1:nsteps
        ρ_joint = U_bg * ρ_joint * U_bg'
        @test _one_site_hilbert_mpo_to_dense(trajectory.states_hilbert[k]) ≈
              _partial_trace_env(ρ_joint, 2, 2) atol=1e-11 rtol=1e-10
    end
end

@testset "process tensor: 1+1 spin TFI PT vs identical split ED" begin
    sys_phys = siteinds("S=1/2", 1)
    env_phys = siteinds("S=1/2", 1)
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
    pt = build_process_tensor(
        system;
        environment=bath,
        dt,
        nsteps,
        sys_alg=Trotter{1}(),
    )

    rho_sys0_h = to_dm(MPS(sys_phys, ["Up"]))
    trajectory = evolve(pt, rho_sys0_h)

    H_bg = OpSum()
    H_bg += 1.0, "Sx", 2
    H_bg += 1.0, "Sz", 1, "Sz", 2
    joint_sites = _joint_phys_sites(sys_phys, env_phys)
    rho_joint = kron(
        hilbert_mpo_to_dense(rho_env0_h, env_phys),
        hilbert_mpo_to_dense(rho_sys0_h, sys_phys),
    )
    U_bg = _exact_unitary_exp(H_bg, joint_sites, dt)
    U_sys = _exact_unitary_exp(H_sys, sys_phys, dt)

    @test length(trajectory.states_liouville) == nsteps
    for k in 1:nsteps
        rho_joint = U_bg * rho_joint * U_bg'
        rho_joint = _apply_system_unitary_on_joint(rho_joint, U_sys, 2)
        @test _one_site_hilbert_mpo_to_dense(trajectory.states_hilbert[k]) ≈
              _partial_trace_env(rho_joint, 2, 2) atol=1e-11 rtol=1e-10
    end
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
    bath = @test_logs (:warn, r"SpinBath: no mode-system coupling") spin_bath([mode])

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

    system = @test_logs (:warn, r"SpinSystem: H is empty") spin_system(sys_phys, OpSum())
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
        hilbert_mpo_to_dense(rho_env0_h, env_phys),
        hilbert_mpo_to_dense(rho_sys0_h, sys_phys),
    )

    # This invariant trajectory is useful, but is not a bath-chronology guard.
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
        hilbert_mpo_to_dense(rho_env0_h, env_phys),
        hilbert_mpo_to_dense(rho_sys0_h, sys_phys),
    )

    grids = ((4, 0.1), (8, 0.05), (16, 0.025))
    t_final = 0.4
    errs = Float64[]
    for (nsteps, dt) in grids
        @test isapprox(nsteps * dt, t_final; atol=1e-12)
        pt = build_process_tensor(
            system;
            environment=bath,
            dt,
            nsteps,
            sys_alg=Trotter{1}(),
        )
        trj = evolve(pt, rho_sys0_h)
        rho_split = _partial_trace_env(
            _evolve_joint_split_exact(
                rho_joint,
                dt,
                nsteps,
                H_sys,
                H_bg,
                sys_phys,
                joint_sites;
                denv=2,
            ),
            2,
            2,
        )
        @test _one_site_hilbert_mpo_to_dense(trj.states_hilbert[end]) ≈
              rho_split atol=1e-11 rtol=1e-10
        rho_full = _reduced_system_joint_full(
            rho_joint,
            t_final,
            H_full,
            joint_sites,
            2,
            2,
        )
        push!(errs, norm(rho_split - rho_full))
    end
    @test all(errs[2:end] .< errs[1:(end - 1)])
    @test all(0.7 .< log2.(errs[1:(end - 1)] ./ errs[2:end]) .< 1.3)
end

@testset "process tensor: empty H_sys diagonal Sz coupling leaves |Up> invariant" begin
    sys_phys = siteinds("S=1/2", 1)
    env_phys = siteinds("S=1/2", 1)
    env_liouv = liouv_sites(env_phys)

    system = @test_logs (:warn, r"SpinSystem: H is empty") spin_system(sys_phys, OpSum())
    rho_env_l = to_liouville(to_dm(MPS(env_phys, ["Up"])); sites=env_liouv)
    cpl = OpSum() + (0.25, "Sz", 1, "Sz", 2)
    mode = @test_logs (:warn, r"SpinMode:H is empty") spin_mode(
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

@testset "process tensor: sys_alg Trotter{2} vs Trotter{1} vs joint ED" begin
    # Coarse Δt so first-order splitting error is visible; fixed total time T.
    # Tolerance: err(Trotter{2}) ≤ 0.1 * err(Trotter{1}) (one decimal-order gain).
    sys_phys = siteinds("S=1/2", 1)
    env_phys = siteinds("S=1/2", 1)
    sys_liouv = liouv_sites(sys_phys)
    env_liouv = liouv_sites(env_phys)

    H_sys = OpSum() + (1.2, "Sx", 1)
    system = spin_system(sys_phys, H_sys)

    rho_env0_h = to_dm(MPS(env_phys, ["Up"]))
    rho_env0_l = to_liouville(rho_env0_h; sites=env_liouv)
    H_env = OpSum() + (0.8, "Sx", 1)
    cpl = OpSum() + (1.5, "Sz", 1, "Sz", 2)
    bath = spin_bath([spin_mode(env_liouv, H_env, rho_env0_l; coupling=cpl)])

    dt = 0.25
    nsteps = 4
    # Each PT slab embeds one full timestep; after `nsteps` slabs the physical time is
    # `nsteps * dt` (evolve's time labels are offset by one step — compare to ED at the
    # physical duration).
    T = nsteps * dt
    rho0_h = to_dm(MPS(sys_phys, ["Up"]))

    @test_throws ArgumentError build_process_tensor(
        system; environment=bath, dt=dt, nsteps=nsteps, sys_alg=Exact(),
    )

    pt1 = build_process_tensor(
        system; environment=bath, dt=dt, nsteps=nsteps, alg=Exact(), sys_alg=Trotter{1}(),
    )
    pt2 = build_process_tensor(
        system; environment=bath, dt=dt, nsteps=nsteps, alg=Exact(), sys_alg=Trotter{2}(),
    )

    trj1 = evolve(pt1, rho0_h)
    trj2 = evolve(pt2, rho0_h)

    H_bg = OpSum() + (0.8, "Sx", 2) + (1.5, "Sz", 1, "Sz", 2)
    joint_sites = _joint_phys_sites(sys_phys, env_phys)
    H_full = _build_joint_full_opsum(H_sys, H_bg)
    rho_joint0 = kron(
        hilbert_mpo_to_dense(rho_env0_h, env_phys),
        hilbert_mpo_to_dense(rho0_h, sys_phys),
    )
    U_bg = _exact_unitary_exp(H_bg, joint_sites, dt)
    U_sys = _exact_unitary_exp(H_sys, sys_phys, dt)
    U_half = _exact_unitary_exp(H_sys, sys_phys, dt / 2)
    rho_split1 = copy(rho_joint0)
    rho_split2 = copy(rho_joint0)
    for _ in 1:nsteps
        rho_split1 = U_bg * rho_split1 * U_bg'
        rho_split1 = _apply_system_unitary_on_joint(rho_split1, U_sys, 2)
        rho_split2 = _apply_system_unitary_on_joint(rho_split2, U_half, 2)
        rho_split2 = U_bg * rho_split2 * U_bg'
        rho_split2 = _apply_system_unitary_on_joint(rho_split2, U_half, 2)
    end
    rho_split1 = _partial_trace_env(rho_split1, 2, 2)
    rho_split2 = _partial_trace_env(rho_split2, 2, 2)
    rho1 = _one_site_hilbert_mpo_to_dense(trj1.states_hilbert[end])
    rho2 = _one_site_hilbert_mpo_to_dense(trj2.states_hilbert[end])
    @test rho1 ≈ rho_split1 atol=1e-11 rtol=1e-10
    @test rho2 ≈ rho_split2 atol=1e-11 rtol=1e-10

    rho_ed = _partial_trace_env(
        _evolve_joint_full_exact(rho_joint0, T, H_full, joint_sites),
        2,
        2,
    )

    err1 = norm(rho1 - rho_ed)
    err2 = norm(rho2 - rho_ed)
    @test err1 > 1e-4  # coarse Δt: asymmetric error must be nontrivial
    @test err2 ≤ 0.1 * err1
end
