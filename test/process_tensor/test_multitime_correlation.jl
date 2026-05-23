# Process-tensor two-time correlators ⟨A(t_A) B(t_B)⟩ vs split joint ED.
using ProcessTensors
using ITensors
using Test
using LinearAlgebra

if !isdefined(Main, :hilbert_matrix_to_mpo)
    include(joinpath(@__DIR__, "..", "time_evolution", "tebd_test_utils.jl"))
end
if !isdefined(Main, :_joint_phys_sites)
    include(joinpath(@__DIR__, "pt_ed_test_utils.jl"))
end

function _assert_corr_matches(val_pt, val_ed)
    @test val_pt isa ComplexF64
    @test isapprox(val_pt, val_ed; atol=_PT_SPLIT_CORR_ATOL, rtol=_PT_SPLIT_CORR_RTOL)
end

@testset "process tensor: two_time_correlation_seq vs joint ED" begin
    sys_phys = siteinds("S=1/2", 1)
    env_phys = siteinds("S=1/2", 1)
    env_liouv = liouv_sites(env_phys)
    dt = 0.05
    default_instr = IdentityOperation()

    H_sys = OpSum()
    H_sys += 1.0, "Sx", 1
    system = spin_system(sys_phys, H_sys)
    rho_sys0_h = to_dm(MPS(sys_phys, ["Dn"]))
    rho_env0_h = to_dm(MPS(env_phys, ["Up"]))
    rho_env0_l = to_liouville(rho_env0_h; sites=env_liouv)
    H_env = OpSum()
    H_env += 1.0, "Sx", 1
    coupling = OpSum()
    coupling += 1.0, "Sz", 1, "Sz", 2
    bath = spin_bath([spin_mode(env_liouv, H_env, rho_env0_l; coupling=coupling)])
    H_bg = OpSum()
    H_bg += 1.0, "Sx", 2
    H_bg += 1.0, "Sz", 1, "Sz", 2

    O_Sz = OpSum()
    O_Sz += 1.0, "Sz", 1
    O_Sx = OpSum()
    O_Sx += 1.0, "Sx", 1
    O_Sy = OpSum()
    O_Sy += 1.0, "Sy", 1

    @testset "minimal PT, t_B = 0 (n_A > n_B)" begin
        n_late = 3
        pt = build_process_tensor(
            system, system.sites[1];
            environment=bath,
            dt=dt,
            nsteps=n_late + 1,
            embed_system_propagation=true,
        )
        for (label, O_A, O_B) in [
            ("Sz(t)Sz(0)", O_Sz, O_Sz),
            ("Sz(t)Sx(0)", O_Sz, O_Sx),
            ("Sx(t)Sy(0)", O_Sx, O_Sy),
        ]
            @testset "$label" begin
                val_pt = evaluate_process(
                    pt,
                    two_time_correlation_seq(pt, (O_A, n_late), (O_B, 0);
                        rho0=rho_sys0_h,
                        default_instr=default_instr,
                    ),
                )
                val_ed = _ed_corr_two_time(
                    rho_sys0_h, rho_env0_h, O_A, O_B, n_late, 0,
                    dt, H_sys, H_bg, sys_phys, env_phys,
                )
                _assert_corr_matches(val_pt, val_ed)
            end
        end
    end

    @testset "minimal PT, t_B > 0 (n_A > n_B)" begin
        pt = build_process_tensor(
            system, system.sites[1];
            environment=bath,
            dt=dt,
            nsteps=4,
            embed_system_propagation=true,
        )
        val_pt = evaluate_process(
            pt,
            two_time_correlation_seq(pt, (O_Sz, 3), (O_Sx, 1);
                rho0=rho_sys0_h,
                default_instr=default_instr,
            ),
        )
        val_ed = _ed_corr_two_time(
            rho_sys0_h, rho_env0_h, O_Sz, O_Sx, 3, 1,
            dt, H_sys, H_bg, sys_phys, env_phys,
        )
        _assert_corr_matches(val_pt, val_ed)
    end

    @testset "minimal PT, reversed order (n_A < n_B)" begin
        pt = build_process_tensor(
            system, system.sites[1];
            environment=bath,
            dt=dt,
            nsteps=4,
            embed_system_propagation=true,
        )
        val_pt = evaluate_process(
            pt,
            two_time_correlation_seq(pt, (O_Sx, 1), (O_Sz, 3);
                rho0=rho_sys0_h,
                default_instr=default_instr,
            ),
        )
        val_ed = _ed_corr_two_time(
            rho_sys0_h, rho_env0_h, O_Sx, O_Sz, 1, 3,
            dt, H_sys, H_bg, sys_phys, env_phys,
        )
        _assert_corr_matches(val_pt, val_ed)
    end

    @testset "minimal PT, same time (n_A = n_B)" begin
        n_late = 2
        pt = build_process_tensor(
            system, system.sites[1];
            environment=bath,
            dt=dt,
            nsteps=n_late + 1,
            embed_system_propagation=true,
        )
        @test pt.nsteps == n_late + 1
        val_pt = evaluate_process(
            pt,
            two_time_correlation_seq(pt, (O_Sz, 2), (O_Sx, 2);
                rho0=rho_sys0_h,
                default_instr=default_instr,
            ),
        )
        val_ed = _ed_corr_two_time(
            rho_sys0_h, rho_env0_h, O_Sz, O_Sx, 2, 2,
            dt, H_sys, H_bg, sys_phys, env_phys,
        )
        _assert_corr_matches(val_pt, val_ed)
    end

    @testset "extended PT (n_late interior to full horizon)" begin
        n_late = 3
        pt = build_process_tensor(
            system, system.sites[1];
            environment=bath,
            dt=dt,
            nsteps=5,
            embed_system_propagation=true,
        )
        @test pt.nsteps > n_late + 1

        val_pt = evaluate_process(
            pt,
            two_time_correlation_seq(pt, (O_Sz, n_late), (O_Sz, 0);
                rho0=rho_sys0_h,
                default_instr=default_instr,
            ),
        )
        val_ed = _ed_corr_two_time(
            rho_sys0_h, rho_env0_h, O_Sz, O_Sz, n_late, 0,
            dt, H_sys, H_bg, sys_phys, env_phys,
        )
        _assert_corr_matches(val_pt, val_ed)

        val_pt = evaluate_process(
            pt,
            two_time_correlation_seq(pt, (O_Sz, n_late), (O_Sx, 1);
                rho0=rho_sys0_h,
                default_instr=default_instr,
            ),
        )
        val_ed = _ed_corr_two_time(
            rho_sys0_h, rho_env0_h, O_Sz, O_Sx, n_late, 1,
            dt, H_sys, H_bg, sys_phys, env_phys,
        )
        _assert_corr_matches(val_pt, val_ed)
    end
end
