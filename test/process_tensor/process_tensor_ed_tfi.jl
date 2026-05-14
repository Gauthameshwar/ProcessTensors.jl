using ProcessTensors
using ITensors
using Test
using LinearAlgebra

if !isdefined(Main, :liouville_state_to_dense)
    include(joinpath(@__DIR__, "time_evolution", "tebd_test_utils.jl"))
end

function _partial_trace_env(rho::AbstractMatrix{<:Number}, dsys::Int, denv::Int)
    rho4 = reshape(ComplexF64.(rho), dsys, denv, dsys, denv)
    out = zeros(ComplexF64, dsys, dsys)
    for e in 1:denv
        out .+= @view rho4[:, e, :, e]
    end
    return out
end

function _apply_system_unitary_on_joint(
    rho_joint::AbstractMatrix{<:Number},
    U_sys::AbstractMatrix{<:Number},
    denv::Int,
)
    U_joint = kron(Matrix{ComplexF64}(I, denv, denv), U_sys)
    return U_joint * rho_joint * U_joint'
end

function _physical_sites_from_hilbert_mpo(rho::AbstractMPO{Hilbert})
    return [only(filter(i -> plev(i) == 0, inds(rho.core[j]))) for j in 1:length(rho.core)]
end

@testset "process tensor: 1+1 spin TFI instrument+PT split vs dense reference" begin
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
    mode = spin_mode(env_liouv, H_env, rho_env0_l)

    coupling = OpSum()
    coupling += 1.0, "Sz", 1, "Sz", 2
    bath = spin_bath([mode]; coupling=coupling)

    dt = 0.1
    nsteps = 8
    pt = build_process_tensor(system, system.sites[1]; environment=bath, dt=dt, nsteps=nsteps)

    rho_sys0_h = to_dm(MPS(sys_phys, ["Up"]))
    trajectory = evolve(pt, rho_sys0_h)

    H_sys_dense = dense_hamiltonian_matrix(H_sys, sys_phys)
    U_sys = exp(-1im * dt * H_sys_dense)

    H_bg = OpSum()
    H_bg += 1.0, "Sx", 2
    H_bg += 1.0, "Sz", 1, "Sz", 2
    joint_sites = [sys_phys[1], env_phys[1]]
    H_bg_dense = dense_hamiltonian_matrix(H_bg, joint_sites)
    U_bg = exp(-1im * dt * H_bg_dense)

    rho_joint = kron(
        hilbert_mpo_to_dense(rho_sys0_h, sys_phys),
        hilbert_mpo_to_dense(rho_env0_h, env_phys),
    )

    @test length(trajectory.states_liouville) == nsteps
    for k in 0:(nsteps - 1)
        rho_pt_h = to_hilbert(trajectory.states_liouville[k + 1])
        rho_pt = hilbert_mpo_to_dense(rho_pt_h, _physical_sites_from_hilbert_mpo(rho_pt_h))
        rho_ed = _partial_trace_env(rho_joint, 2, 2)
        @test isapprox(rho_pt, rho_ed; atol=1e-10, rtol=1e-8)
        rho_joint = _apply_system_unitary_on_joint(rho_joint, U_sys, 2)
        rho_joint = U_bg * rho_joint * U_bg'
    end
end
