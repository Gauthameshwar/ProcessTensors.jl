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

function _single_site_opsum(opname::AbstractString)
    O = OpSum()
    O += 1.0, opname, 1
    return O
end

function _pt_corr_AtB0(
    pt::ProcessTensor,
    rho_sys0_h,
    O_A::OpSum,
    O_B::OpSum,
    sys_phys;
    default_instr::AbstractInstrument,
)
    rho0 = hilbert_mpo_to_dense(rho_sys0_h, sys_phys)
    B = dense_hamiltonian_matrix(O_B, sys_phys)
    rho_B0_h = hilbert_matrix_to_mpo(B * rho0, sys_phys)

    seq = InstrumentSeq(default=default_instr, nsteps=pt.nsteps)
    add!(seq, StatePreparation(rho_B0_h), 0)
    add!(seq, ObservableMeasurement(O_A), pt.nsteps)
    return evaluate_process(pt, seq; default_instr=default_instr)
end


"""
``⟨A(t) B(0)⟩`` from joint ED: insert `B` at `t = 0`, evolve, then `Tr[A ρ_sys(t)]`.

`evolution = :split` (default) matches the PT instrument schedule
(`SystemPropagation` + bath slabs). `evolution = :full` uses a single
`exp(-im * t * H_full)` built from the joint `OpSum` on `joint_sites`.
"""
function _ed_corr_AtB0(
    rho_sys0_h,
    rho_env0_h,
    O_A::OpSum,
    O_B::OpSum,
    n2::Int,
    dt::Real,
    H_sys::OpSum,
    H_bg::OpSum,
    sys_phys::AbstractVector{<:Index},
    env_phys::AbstractVector{<:Index};
    denv::Int = dim(only(env_phys)),
    evolution::Symbol = :split,
)
    rho_sys0 = hilbert_mpo_to_dense(rho_sys0_h, sys_phys)
    rho_env0 = hilbert_mpo_to_dense(rho_env0_h, env_phys)
    joint_sites = _joint_phys_sites(sys_phys, env_phys)
    H_full = _build_joint_full_opsum(H_sys, H_bg)
    _ = _joint_hamiltonian_dense(H_full, joint_sites)

    rho_joint_B0 = _joint_density_B_at_0(rho_sys0, rho_env0, O_B, sys_phys)
    t = n2 * dt
    rho_joint_t = if evolution === :split
        _evolve_joint_split_exact(
            rho_joint_B0,
            dt,
            n2,
            H_sys,
            H_bg,
            sys_phys,
            joint_sites;
            denv=denv,
        )
    elseif evolution === :full
        _evolve_joint_full_exact(rho_joint_B0, t, H_full, joint_sites)
    else
        throw(ArgumentError("_ed_corr_AtB0: evolution must be :split or :full; got $evolution."))
    end

    dsys = dim(only(sys_phys))
    A = dense_hamiltonian_matrix(O_A, sys_phys)
    rho_sys_t = _partial_trace_env(rho_joint_t, dsys, denv)
    return tr(A * rho_sys_t)
end


@testset "process tensor: multi-time correlations vs joint ED" begin
    sys_phys = siteinds("S=1/2", 1)
    env_phys = siteinds("S=1/2", 1)
    env_liouv = liouv_sites(env_phys)

    H_sys = _single_site_opsum("Sx")
    system = spin_system(sys_phys, H_sys)

    rho_sys0_h = to_dm(MPS(sys_phys, ["Dn"]))
    rho_env0_h = to_dm(MPS(env_phys, ["Up"]))
    rho_env0_l = to_liouville(rho_env0_h; sites=env_liouv)

    H_env = _single_site_opsum("Sx")
    coupling = OpSum() + (1.0, "Sz", 1, "Sz", 2)
    mode = spin_mode(env_liouv, H_env, rho_env0_l; coupling=coupling)
    bath = spin_bath([mode])

    H_bg = OpSum()
    H_bg += 1.0, "Sx", 2
    H_bg += 1.0, "Sz", 1, "Sz", 2

    dt = 0.05
    n2 = 3
    nsteps = n2 + 1
    pt = build_process_tensor(
        system,
        system.sites[1];
        environment=bath,
        dt=dt,
        nsteps=nsteps,
        embed_system_propagation=false,
    )
    default_instr = SystemPropagation(system)

    cases = [
        ("Sz(t)Sz(0)", _single_site_opsum("Sz"), _single_site_opsum("Sz")),
        ("Sz(t)Sx(0)", _single_site_opsum("Sz"), _single_site_opsum("Sx")),
        ("Sx(t)Sy(0)", _single_site_opsum("Sx"), _single_site_opsum("Sy")),
    ]

    for (label, O_A, O_B) in cases
        @testset "$label" begin
            # Correlation from the process tensor
            val_pt = _pt_corr_AtB0(
                pt,
                rho_sys0_h,
                O_A,
                O_B,
                sys_phys;
                default_instr=default_instr,
            )
            # Correlation from the Trotterized joint ED
            val_ed = _ed_corr_AtB0(
                rho_sys0_h,
                rho_env0_h,
                O_A,
                O_B,
                n2,
                dt,
                H_sys,
                H_bg,
                sys_phys,
                env_phys;
                evolution=:split,
            )
            @test val_pt isa ComplexF64
            @test isapprox(val_pt, val_ed; atol=1e-10, rtol=1e-8)

            # Correlation from the exact joint ED
            val_full = _ed_corr_AtB0(
                rho_sys0_h,
                rho_env0_h,
                O_A,
                O_B,
                n2,
                dt,
                H_sys,
                H_bg,
                sys_phys,
                env_phys;
                evolution=:full,
            )
            @test isapprox(val_ed, val_full; atol=0.01, rtol=0.05)
        end
    end
end
