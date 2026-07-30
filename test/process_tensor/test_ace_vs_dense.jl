# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: test/process_tensor/test_ace_vs_dense.jl
# Contributor: Gauthameshwar S.
#
# Tests that sequential ACE process-tensor construction reproduces Dense/ED
# reduced dynamics, preserves causality through compressed memory bonds, and
# compresses bond dimensions under finite cutoffs.
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

if !isdefined(Main, :JULIA_PROCESSTENSORS_RUN_SLOW)
    const JULIA_PROCESSTENSORS_RUN_SLOW = get(ENV, "JULIA_PROCESSTENSORS_RUN_SLOW", "true") == "true"
end

"""Dense one-site density matrices for every `evolve` snapshot."""
function _ace_dense_traj(pt::ProcessTensor, rho0_h)
    trajectory = evolve(pt, rho0_h)
    return [_one_site_liouville_state_to_dense(ρ) for ρ in trajectory.states_liouville]
end

_ace_traj_err(traj_a, traj_b) = maximum(norm(a - b) for (a, b) in zip(traj_a, traj_b))

"""Fresh spin mode on its own Liouville site, coupled to system site 2 by `cpl_op`."""
function _ace_spin_mode(h_coeff::Real, cpl_coeff::Real; cpl_op::AbstractString="Sz")
    env_phys = siteinds("S=1/2", 1)
    env_liouv = liouv_sites(env_phys)
    rho_env_l = to_liouville(to_dm(MPS(env_phys, ["Up"])); sites=env_liouv)
    H_mode = OpSum() + (h_coeff, "Sx", 1)
    cpl = OpSum() + (cpl_coeff, cpl_op, 1, cpl_op, 2)
    return spin_mode(env_liouv, H_mode, rho_env_l; coupling=cpl)
end

@testset "ACE builder: constructor and build guards" begin
    @test_throws ArgumentError ACE(; cutoff=-1e-3)
    @test_throws ArgumentError ACE(; maxdim=0)

    sys_phys = siteinds("S=1/2", 1)
    system = spin_system(sys_phys, OpSum() + (0.3, "Sz", 1))
    bath = spin_bath([_ace_spin_mode(0.5, 0.1), _ace_spin_mode(0.6, 0.12)])

    # Both sys_alg values are accepted for method=ACE().
    for sys_alg in (Trotter{1}(), Trotter{2}())
        pt = build_process_tensor(
            system, system.sites[1];
            method=ACE(cutoff=0.0), environment=bath, dt=0.05, nsteps=2,
            sys_alg=sys_alg,
        )
        @test pt isa ProcessTensor
        @test length(pt.core) == 2
    end

    # Inter-mode/global coupling breaks mode independence and must be rejected.
    bath_coupled = spin_bath(
        [_ace_spin_mode(0.5, 0.1), _ace_spin_mode(0.6, 0.12)];
        coupling=OpSum() + (0.1, "Sz", 1, "Sz", 2),
    )
    @test_throws ArgumentError build_process_tensor(
        system, system.sites[1];
        method=ACE(), environment=bath_coupled, dt=0.05, nsteps=2,
    )

    # combine_alg is restricted to Trotter{1}/Trotter{2}.
    @test_throws ArgumentError build_process_tensor(
        system, system.sites[1];
        method=ACE(), environment=bath, dt=0.05, nsteps=2,
        combine_alg=Exact(),
    )

    # No-environment ACE falls back to the Markovian trivial PT.
    pt_markov = build_process_tensor(
        system, system.sites[1]; method=ACE(), dt=0.05, nsteps=3,
    )
    @test pt_markov isa ProcessTensor
    @test maxlinkdim(pt_markov.core) == 1
end

@testset "ACE ≡ Dense: single spin mode, all sys_alg × combine_alg" begin
    sys_phys = siteinds("S=1/2", 1)
    system = spin_system(sys_phys, OpSum() + (0.3, "Sz", 1) + (0.2, "Sx", 1))
    bath = spin_bath([_ace_spin_mode(0.5, 0.4)])
    rho0_h = to_dm(MPS(sys_phys, ["Up"]))
    dt = 0.1
    nsteps = 5

    for sys_alg in (Trotter{1}(), Trotter{2}())
        pt_dense = build_process_tensor(
            system, system.sites[1];
            method=Dense(), environment=bath, dt=dt, nsteps=nsteps, sys_alg=sys_alg,
        )
        traj_dense = _ace_dense_traj(pt_dense, rho0_h)
        for combine_alg in (Trotter{1}(), Trotter{2}())
            pt_ace = build_process_tensor(
                system, system.sites[1];
                method=ACE(cutoff=0.0), environment=bath, dt=dt, nsteps=nsteps,
                sys_alg=sys_alg, combine_alg=combine_alg,
            )
            validate_process_tensor_structure(pt_ace)
            traj_ace = _ace_dense_traj(pt_ace, rho0_h)
            @test _ace_traj_err(traj_ace, traj_dense) < 1e-10
            for ρ in traj_ace
                @test abs(real(tr(ρ)) - 1.0) < 1e-9
            end
        end
    end
end

@testset "ACE ≡ Dense: two commuting spin modes and mode-order invariance" begin
    sys_phys = siteinds("S=1/2", 1)
    system = spin_system(sys_phys, OpSum() + (0.3, "Sz", 1) + (0.2, "Sx", 1))
    m1 = _ace_spin_mode(0.5, 0.4)
    m2 = _ace_spin_mode(0.7, 0.5)
    bath = spin_bath([m1, m2])
    rho0_h = to_dm(MPS(sys_phys, ["Up"]))
    dt = 0.1
    nsteps = 6

    pt_dense = build_process_tensor(
        system, system.sites[1]; method=Dense(), environment=bath, dt=dt, nsteps=nsteps,
    )
    traj_dense = _ace_dense_traj(pt_dense, rho0_h)

    # Both modes couple through Sz⊗Sz, so the mode Liouvillians commute and the
    # sequential mode splitting introduces no Trotter error: ACE must match
    # Dense to numerical precision for both combine styles and either order.
    for combine_alg in (Trotter{1}(), Trotter{2}())
        pt_ace = build_process_tensor(
            system, system.sites[1];
            method=ACE(cutoff=0.0), environment=bath, dt=dt, nsteps=nsteps,
            combine_alg=combine_alg,
        )
        validate_process_tensor_structure(pt_ace)
        @test _ace_traj_err(_ace_dense_traj(pt_ace, rho0_h), traj_dense) < 1e-10

        bath_swapped = spin_bath([m2, m1])
        pt_swapped = build_process_tensor(
            system, system.sites[1];
            method=ACE(cutoff=0.0), environment=bath_swapped, dt=dt, nsteps=nsteps,
            combine_alg=combine_alg,
        )
        @test _ace_traj_err(_ace_dense_traj(pt_swapped, rho0_h), traj_dense) < 1e-10
    end
end

@testset "ACE vs joint ED: two non-commuting spin modes" begin
    sys_phys = siteinds("S=1/2", 1)
    H_sys = OpSum() + (0.7, "Sx", 1)
    system = spin_system(sys_phys, H_sys)

    h1, h2 = 0.3, 0.35
    g1, g2 = 0.04, 0.05
    m1 = _ace_spin_mode(h1, g1; cpl_op="Sz")
    m2 = _ace_spin_mode(h2, g2; cpl_op="Sx")
    bath = spin_bath([m1, m2])
    rho0_h = to_dm(MPS(sys_phys, ["Up"]))
    dt = 0.1
    nsteps = 6

    # Joint ED reference on [system, mode1, mode2].
    env_phys = siteinds("S=1/2", 2)
    joint_sites = _joint_phys_sites(sys_phys, env_phys)
    H_bg = OpSum()
    H_bg += h1, "Sx", 2
    H_bg += g1, "Sz", 2, "Sz", 1
    H_bg += h2, "Sx", 3
    H_bg += g2, "Sx", 3, "Sx", 1
    H_full = _build_joint_full_opsum(H_sys, H_bg)
    rho_joint = _joint_initial_density(sys_phys, env_phys)
    denv = 4

    pt_dense = build_process_tensor(
        system, system.sites[1]; method=Dense(), environment=bath, dt=dt, nsteps=nsteps,
    )
    traj_dense = _ace_dense_traj(pt_dense, rho0_h)

    for combine_alg in (Trotter{1}(), Trotter{2}())
        pt_ace = build_process_tensor(
            system, system.sites[1];
            method=ACE(cutoff=0.0), environment=bath, dt=dt, nsteps=nsteps,
            combine_alg=combine_alg,
        )
        traj_ace = _ace_dense_traj(pt_ace, rho0_h)

        # ACE differs from Dense only by the sequential mode-splitting Trotter
        # error, which is small for these weak couplings.
        @test _ace_traj_err(traj_ace, traj_dense) < 1e-3

        joint_errs = Float64[]
        for k in 0:(nsteps - 1)
            rho_ed = _reduced_system_joint_full(rho_joint, k * dt, H_full, joint_sites, 2, denv)
            push!(joint_errs, norm(traj_ace[k + 1] - rho_ed))
        end
        @test maximum(joint_errs) < 0.05
    end
end

@testset "ACE: uncoupled modes reproduce the free-system trajectory" begin
    sys_phys = siteinds("S=1/2", 1)
    system = spin_system(sys_phys, OpSum() + (0.4, "Sz", 1) + (0.3, "Sx", 1))
    rho0_h = to_dm(MPS(sys_phys, ["Up"]))
    dt = 0.05
    nsteps = 4

    modes = SpinMode[]
    for m in 1:2
        env_phys = siteinds("S=1/2", 1)
        env_liouv = liouv_sites(env_phys)
        rho_env_l = to_liouville(to_dm(MPS(env_phys, ["Up"])); sites=env_liouv)
        push!(modes, spin_mode(env_liouv, OpSum() + (0.25 + 0.1 * m, "Sx", 1), rho_env_l))
    end
    bath = @test_warn r"SpinBath: no mode-system coupling" spin_bath(modes)

    pt_free = build_process_tensor(system, system.sites[1]; dt=dt, nsteps=nsteps)
    pt_ace = build_process_tensor(
        system, system.sites[1];
        method=ACE(cutoff=0.0), environment=bath, dt=dt, nsteps=nsteps,
    )
    @test _ace_traj_err(_ace_dense_traj(pt_ace, rho0_h), _ace_dense_traj(pt_free, rho0_h)) < 1e-10
end

@testset "ACE compression: finite cutoff lowers χ, tighter cutoff lowers error" begin
    sys_phys = siteinds("S=1/2", 1)
    system = spin_system(sys_phys, OpSum() + (0.3, "Sz", 1) + (0.2, "Sx", 1))
    bath = spin_bath([_ace_spin_mode(0.5, 0.4), _ace_spin_mode(0.7, 0.5)])
    rho0_h = to_dm(MPS(sys_phys, ["Up"]))
    dt = 0.1
    nsteps = 6

    pt_dense = build_process_tensor(
        system, system.sites[1]; method=Dense(), environment=bath, dt=dt, nsteps=nsteps,
    )
    traj_dense = _ace_dense_traj(pt_dense, rho0_h)

    build_ace(cutoff; maxdim=typemax(Int)) = build_process_tensor(
        system, system.sites[1];
        method=ACE(cutoff=cutoff, maxdim=maxdim), environment=bath,
        dt=dt, nsteps=nsteps,
    )

    pt_exact = build_ace(0.0)
    pt_tight = build_ace(1e-10)
    pt_loose = build_ace(1e-6)

    χ_exact = maxlinkdim(pt_exact.core)
    @test χ_exact == 16
    @test maxlinkdim(pt_loose.core) < χ_exact
    @test maxlinkdim(pt_tight.core) <= χ_exact

    err_exact = _ace_traj_err(_ace_dense_traj(pt_exact, rho0_h), traj_dense)
    err_tight = _ace_traj_err(_ace_dense_traj(pt_tight, rho0_h), traj_dense)
    err_loose = _ace_traj_err(_ace_dense_traj(pt_loose, rho0_h), traj_dense)

    @test err_exact < 1e-10
    @test err_loose < 1e-3
    @test err_tight <= err_loose + 1e-12

    # maxdim cap alone also compresses while staying accurate here.
    pt_capped = build_ace(0.0; maxdim=4)
    @test maxlinkdim(pt_capped.core) == 4
    @test _ace_traj_err(_ace_dense_traj(pt_capped, rho0_h), traj_dense) < 1e-6

    # Compressed marginals stay physical.
    for ρ in _ace_dense_traj(pt_loose, rho0_h)
        _assert_hermitian_psd(ρ; atol=1e-5)
        _assert_trace_one(ρ; atol=1e-5)
    end
end

@testset "ACE causality: prefix PTs reproduce compressed marginals" begin
    sys_phys = siteinds("S=1/2", 1)
    system = spin_system(sys_phys, OpSum() + (0.7, "Sx", 1))
    bath = spin_bath([_ace_spin_mode(0.3, 0.2; cpl_op="Sz"), _ace_spin_mode(0.35, 0.25; cpl_op="Sx")])
    rho0_h = to_dm(MPS(sys_phys, ["Up"]))
    dt = 0.1
    nsteps = 5

    for combine_alg in (Trotter{1}(), Trotter{2}())
        pt_full = build_process_tensor(
            system, system.sites[1];
            method=ACE(cutoff=1e-12), environment=bath, dt=dt, nsteps=nsteps,
            combine_alg=combine_alg,
        )
        traj_full = _ace_dense_traj(pt_full, rho0_h)

        # The marginal at snapshot k must not depend on the process tensor's
        # future length: a PT truncated to k+1 steps (whose final snapshot needs
        # no memory-bond closure) must agree with the closure-based snapshot of
        # the full PT.
        for k in 0:(nsteps - 1)
            pt_k = build_process_tensor(
                system, system.sites[1];
                method=ACE(cutoff=1e-12), environment=bath, dt=dt, nsteps=k + 1,
                combine_alg=combine_alg,
            )
            traj_k = _ace_dense_traj(pt_k, rho0_h)
            @test norm(traj_full[k + 1] - traj_k[k + 1]) < 1e-8
        end
    end
end

if JULIA_PROCESSTENSORS_RUN_SLOW
    @testset "ACE vs joint ED: nmodes=3 spin star bath [slow]" begin
        nmodes = 3
        sys_phys = siteinds("S=1/2", 1)
        env_phys = siteinds("S=1/2", nmodes)
        env_liouv = liouv_sites(env_phys)

        H_sys = OpSum() + (0.7, "Sx", 1)
        system = spin_system(sys_phys, H_sys)

        mode_h_coeffs = [0.25 + 0.1 * m for m in 1:nmodes]
        mode_cpl_coeffs = [0.03 + 0.01 * m for m in 1:nmodes]

        modes = SpinMode[]
        for m in 1:nmodes
            rho_env_l = to_liouville(to_dm(MPS([env_phys[m]], ["Up"])); sites=[env_liouv[m]])
            H_mode = OpSum() + (mode_h_coeffs[m], "Sx", 1)
            cpl_mode = OpSum() + (mode_cpl_coeffs[m], "Sz", 1, "Sz", 2)
            push!(modes, spin_mode([env_liouv[m]], H_mode, rho_env_l; coupling=cpl_mode))
        end
        bath = spin_bath(modes)

        dt = 0.1
        nsteps = 6
        rho0_h = to_dm(MPS(sys_phys, ["Up"]))

        denv = 2^nmodes
        joint_sites = _joint_phys_sites(sys_phys, env_phys)
        H_bg = _build_multimode_bath_opsum(nmodes, mode_h_coeffs, mode_cpl_coeffs)
        H_full = _build_joint_full_opsum(H_sys, H_bg)
        rho_joint = _joint_initial_density(sys_phys, env_phys)

        for combine_alg in (Trotter{1}(), Trotter{2}())
            pt_ace = build_process_tensor(
                system, system.sites[1];
                method=ACE(cutoff=1e-12), environment=bath, dt=dt, nsteps=nsteps,
                combine_alg=combine_alg,
            )
            validate_process_tensor_structure(pt_ace)
            traj_ace = _ace_dense_traj(pt_ace, rho0_h)

            joint_errs = Float64[]
            trace_errs = Float64[]
            for k in 0:(nsteps - 1)
                rho_ed = _reduced_system_joint_full(rho_joint, k * dt, H_full, joint_sites, 2, denv)
                push!(joint_errs, norm(traj_ace[k + 1] - rho_ed))
                push!(trace_errs, abs(real(tr(traj_ace[k + 1])) - 1.0))
            end
            @test maximum(joint_errs) < 0.05
            @test maximum(trace_errs) < 1e-8
        end
    end

    @testset "ACE ≡ Dense: tiny bosonic baths [slow]" begin
        sys_phys = siteinds("S=1/2", 1)
        system = spin_system(sys_phys, OpSum() + (0.5, "Sz", 1) + (0.2, "Sx", 1))
        rho0_h = to_dm(MPS(sys_phys, ["Up"]))
        dt = 0.1
        nsteps = 4

        function _ace_boson_mode(nlevels::Int, ω::Real, g::Real)
            b_phys = siteinds("Boson", 1; dim=nlevels)
            b_liouv = liouv_sites(b_phys)
            rho_b = to_liouville(to_dm(MPS(b_phys, ["0"])); sites=b_liouv)
            H_b = OpSum() + (ω, "N", 1)
            # Displacement-type coupling g (a + a†) Sz.
            cpl = OpSum() + (g, "A", 1, "Sz", 2) + (g, "Adag", 1, "Sz", 2)
            return bosonic_mode(b_liouv, H_b, rho_b; coupling=cpl)
        end

        # Single boson mode: ACE must match Dense for both combine styles.
        bath1 = bosonic_bath([_ace_boson_mode(4, 0.8, 0.3)])
        pt_dense1 = build_process_tensor(
            system, system.sites[1]; method=Dense(), environment=bath1, dt=dt, nsteps=nsteps,
        )
        traj_dense1 = _ace_dense_traj(pt_dense1, rho0_h)
        for combine_alg in (Trotter{1}(), Trotter{2}())
            pt_ace1 = build_process_tensor(
                system, system.sites[1];
                method=ACE(cutoff=0.0), environment=bath1, dt=dt, nsteps=nsteps,
                combine_alg=combine_alg,
            )
            @test _ace_traj_err(_ace_dense_traj(pt_ace1, rho0_h), traj_dense1) < 1e-10
        end

        # Two small boson modes with commuting Sz-displacement couplings.
        bath2 = bosonic_bath([_ace_boson_mode(3, 0.8, 0.3), _ace_boson_mode(3, 1.1, 0.25)])
        pt_dense2 = build_process_tensor(
            system, system.sites[1]; method=Dense(), environment=bath2, dt=dt, nsteps=nsteps,
        )
        traj_dense2 = _ace_dense_traj(pt_dense2, rho0_h)
        pt_ace2 = build_process_tensor(
            system, system.sites[1];
            method=ACE(cutoff=0.0), environment=bath2, dt=dt, nsteps=nsteps,
            combine_alg=Trotter{2}(),
        )
        @test _ace_traj_err(_ace_dense_traj(pt_ace2, rho0_h), traj_dense2) < 1e-8

        # Finite cutoff compresses the boson memory bond while staying accurate.
        pt_ace2c = build_process_tensor(
            system, system.sites[1];
            method=ACE(cutoff=1e-8), environment=bath2, dt=dt, nsteps=nsteps,
            combine_alg=Trotter{2}(),
        )
        @test maxlinkdim(pt_ace2c.core) < maxlinkdim(pt_ace2.core)
        @test _ace_traj_err(_ace_dense_traj(pt_ace2c, rho0_h), traj_dense2) < 1e-4
    end
end
