# Multi-mode spin bath: split PT vs exact joint exp(-im * t * H_full).
# Each mode.coupling uses local OpSum sites 1=bath, 2=system.
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


@testset "process_tensor: bath-only evolution traces out to unchanged system" begin
    nmodes = 2
    sys_phys = siteinds("S=1/2", 1)
    env_phys = siteinds("S=1/2", nmodes)
    env_liouv = liouv_sites(env_phys)

    # Bath-only reference: the system should stay unchanged.
    system = @test_warn r"SpinSystem: H is empty" spin_system(sys_phys, OpSum())
    modes = SpinMode[]

    for m in 1:nmodes
        rho_env_h = to_dm(MPS([env_phys[m]], ["Up"]))
        rho_env_l = to_liouville(rho_env_h; sites=[env_liouv[m]])
        H_mode = OpSum()
        H_mode += 0.25 + 0.1 * m, "Sx", 1
        push!(modes, spin_mode([env_liouv[m]], H_mode, rho_env_l))
    end

    # No coupling means the system density matrix should not change.
    bath = @test_warn r"SpinBath: no mode-system coupling" spin_bath(modes)

    pt = build_process_tensor(system, system.sites[1]; environment=bath, dt=0.05, nsteps=2)
    trajectory = evolve(pt, to_dm(MPS(sys_phys, ["Up"])))
    rho_up = hilbert_mpo_to_dense(to_dm(MPS(sys_phys, ["Up"])), sys_phys)

    for rho_l in trajectory.states_liouville
        rho_h = to_hilbert(rho_l)
        rho_pt = hilbert_mpo_to_dense(rho_h, _physical_sites_from_hilbert_mpo(rho_h))
        @test isapprox(rho_pt, rho_up; atol=1e-9, rtol=1e-7)
    end
end

for nmodes in (2, 3)
    @testset "process_tensor: nontrivial nmodes=$nmodes PT vs joint full-H exact ED" begin
        sys_phys = siteinds("S=1/2", 1)
        env_phys = siteinds("S=1/2", nmodes)
        env_liouv = liouv_sites(env_phys)

        H_sys = OpSum()
        H_sys += 0.7, "Sx", 1
        system = spin_system(sys_phys, H_sys)

        mode_h_coeffs = [0.25 + 0.1 * m for m in 1:nmodes]
        mode_cpl_coeffs = [0.03 + 0.01 * m for m in 1:nmodes]

        modes = SpinMode[]
        for m in 1:nmodes
            rho_env_h = to_dm(MPS([env_phys[m]], ["Up"]))
            rho_env_l = to_liouville(rho_env_h; sites=[env_liouv[m]])
            H_mode = OpSum()
            H_mode += mode_h_coeffs[m], "Sx", 1
            # Star coupling: mode m talks only to the system site.
            cpl_mode = OpSum()
            cpl_mode += mode_cpl_coeffs[m], "Sz", 1, "Sz", 2
            mode = spin_mode([env_liouv[m]], H_mode, rho_env_l; coupling=cpl_mode)
            @test mode.coupling != OpSum()
            push!(modes, mode)
        end
        bath = spin_bath(modes)
        @test length(bath.modes) == nmodes

        dt = 0.1
        nsteps = 6
        pt = build_process_tensor(system, system.sites[1]; environment=bath, dt=dt, nsteps=nsteps)
        @test pt isa ProcessTensor
        @test length(pt.core) == nsteps

        trajectory = evolve(pt, to_dm(MPS(sys_phys, ["Up"])))
        @test length(trajectory.states_liouville) == nsteps

        denv = 2^nmodes
        joint_sites = _joint_phys_sites(sys_phys, env_phys)
        H_bg = _build_multimode_bath_opsum(nmodes, mode_h_coeffs, mode_cpl_coeffs)
        H_full = _build_joint_full_opsum(H_sys, H_bg)
        rho_joint = _joint_initial_density(sys_phys, env_phys)

        joint_errs = Float64[]
        for k in 0:(nsteps - 1)
            t = k * dt
            rho_ed = _reduced_system_joint_full(rho_joint, t, H_full, joint_sites, 2, denv)

            rho_l = trajectory.states_liouville[k + 1]
            rho_h = to_hilbert(rho_l)
            rho_pt = hilbert_mpo_to_dense(rho_h, _physical_sites_from_hilbert_mpo(rho_h))
            push!(joint_errs, norm(rho_pt - rho_ed))
        end
        @test maximum(joint_errs) < 0.2
    end
end

@testset "process_tensor: diagonal multimode coupling keeps |Up><Up| invariant" begin
    nmodes = 3
    sys_phys = siteinds("S=1/2", 1)
    env_phys = siteinds("S=1/2", nmodes)
    env_liouv = liouv_sites(env_phys)

    # With no system Hamiltonian, |Up><Up| should stay fixed.
    system = @test_warn r"SpinSystem: H is empty" spin_system(sys_phys, OpSum())
    modes = SpinMode[]
    for m in 1:nmodes
        rho_env_h = to_dm(MPS([env_phys[m]], ["Up"]))
        rho_env_l = to_liouville(rho_env_h; sites=[env_liouv[m]])
        # Diagonal star coupling keeps the system state unchanged here.
        cpl_mode = OpSum()
        cpl_mode += 0.08 + 0.01 * m, "Sz", 1, "Sz", 2
        mode = @test_warn r"SpinMode:H is empty" spin_mode(
            [env_liouv[m]], OpSum(), rho_env_l; coupling=cpl_mode,
        )
        push!(modes, mode)
    end
    bath = spin_bath(modes)
    pt = build_process_tensor(system, system.sites[1]; environment=bath, dt=0.05, nsteps=5)
    trajectory = evolve(pt, to_dm(MPS(sys_phys, ["Up"])))

    rho_up = hilbert_mpo_to_dense(to_dm(MPS(sys_phys, ["Up"])), sys_phys)
    for rho_l in trajectory.states_liouville
        rho_h = to_hilbert(rho_l)
        rho_pt = hilbert_mpo_to_dense(rho_h, _physical_sites_from_hilbert_mpo(rho_h))
        @test isapprox(rho_pt, rho_up; atol=1e-6, rtol=1e-6)
    end
end
