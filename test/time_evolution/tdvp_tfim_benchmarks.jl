using ProcessTensors
using ITensors
using LinearAlgebra
using Test

if !isdefined(Main, :dense_hamiltonian_matrix)
    include(joinpath(@__DIR__, "tebd_test_utils.jl"))
end

function tfim_hamiltonian(N::Int; J::Float64=1.0, h::Float64=1.2)
    os_H = OpSum()
    for j in 1:(N - 1)
        os_H += -J, "Z", j, "Z", j + 1
    end
    for j in 1:N
        os_H += -h, "X", j
    end
    return os_H
end

tfim_decay_jump_ops(N::Int; γ::Float64=0.5) = [(γ, "S-", j) for j in 1:N]

function dense_one_site_operator(op_name::AbstractString, physical_sites, site::Int)
    local_ops = Matrix{ComplexF64}[]
    for (j, s) in enumerate(physical_sites)
        if j == site
            push!(local_ops, Array(op(op_name, s), prime(s), s))
        else
            push!(local_ops, Matrix{ComplexF64}(I, dim(s), dim(s)))
        end
    end
    O = local_ops[1]
    for j in 2:length(local_ops)
        O = kron(O, local_ops[j])
    end
    return O
end

function average_observable_dense(ρ::AbstractMatrix{<:Number}, embedded_ops)
    Ns = length(embedded_ops)
    return real(sum(LinearAlgebra.tr(ρ * O) for O in embedded_ops) / Ns)
end

@testset "tdvp.jl: Liouville TFIM benchmarks" begin
    N = 4
    physical_sites = siteinds("S=1/2", N)
    liouv_sites_shared = liouv_sites(physical_sites)
    trace_bra = vectorized_identity_state(physical_sites, liouv_sites_shared)
    os_H = tfim_hamiltonian(N; J=1.0, h=1.2)
    H_mpo = MPO(os_H, physical_sites)
    H_dense = dense_hamiltonian_matrix(os_H, physical_sites)
    jump_ops = tfim_decay_jump_ops(N; γ=0.5)

    ψ0 = MPS(physical_sites, fill("Up", N))
    ρ0 = to_dm(ψ0)
    ρ0_vec = to_liouville(ρ0; sites=liouv_sites_shared)
    ρ0_dense = hilbert_mpo_to_dense(ρ0, physical_sites)
    vec0 = vec(ComplexF64.(ρ0_dense))

    L_unitary_mpo = MPO_Liouville(os_H, liouv_sites_shared; jump_ops=Tuple{Number, String, Int}[])
    L_diss_mpo = MPO_Liouville(os_H, liouv_sites_shared; jump_ops=jump_ops)
    L_unitary_dense = dense_liouvillian_matrix(os_H, Tuple{Number, String, Int}[], physical_sites, liouv_sites_shared)
    L_diss_dense = dense_liouvillian_matrix(os_H, jump_ops, physical_sites, liouv_sites_shared)

    @testset "wrapped tdvp forwards time-dependent signature" begin
        state_1 = tdvp(L_unitary_mpo, 0.05, ρ0_vec; time_step=0.05, nsite=1, maxdim=32, cutoff=1e-10, outputlevel=0)
        state_2 = tdvp(L_unitary_mpo, 0.05, ρ0_vec; time_step=0.05, nsite=2, maxdim=32, cutoff=1e-10, outputlevel=0)

        @test state_1 isa MPS{Liouville}
        @test state_2 isa MPS{Liouville}
        @test maxlinkdim(state_2) ≥ maxlinkdim(state_1)
    end

    @testset "dynamic helper switches from 2-site to 1-site" begin
        dynamic = dynamic_tdvp_trajectory(
            ρ0_vec,
            L_unitary_mpo,
            0.05,
            4;
            maxdim=32,
            cutoff=1e-10,
            plateau_patience=1,
            switch_maxdim=2,
        )

        @test dynamic.nsites[1] == 2
        @test any(==(1), dynamic.nsites)
        @test dynamic.bond_dims[end] ≤ 32
    end

    @testset "small closed TFIM: TDVP preserves invariants and follows ED trends" begin
        T = 0.2
        dt = 0.05
        nsteps = round(Int, T / dt)
        x_mpos = single_site_pauli_mpos("X", physical_sites)
        z_mpos = single_site_pauli_mpos("Z", physical_sites)
        x_ops = [dense_one_site_operator("X", physical_sites, j) for j in 1:N]
        z_ops = [dense_one_site_operator("Z", physical_sites, j) for j in 1:N]

        states_1 = tdvp_trajectory(ρ0_vec, L_unitary_mpo, dt, nsteps; nsite=1, maxdim=128, cutoff=1e-10)
        states_2 = tdvp_trajectory(ρ0_vec, L_unitary_mpo, dt, nsteps; nsite=2, maxdim=128, cutoff=1e-10)
        states_dynamic = dynamic_tdvp_trajectory(
            ρ0_vec,
            L_unitary_mpo,
            dt,
            nsteps;
            maxdim=128,
            cutoff=1e-10,
            plateau_patience=1,
            switch_maxdim=4,
        ).states

        exact_dense = reshape(exp(T * L_unitary_dense) * vec0, size(ρ0_dense)...)
        ρ_1 = liouville_state_to_dense(last(states_1), physical_sites)
        ρ_2 = liouville_state_to_dense(last(states_2), physical_sites)
        ρ_dynamic = liouville_state_to_dense(last(states_dynamic), physical_sites)

        err_1 = relative_frobenius_error(ρ_1, exact_dense)
        err_2 = relative_frobenius_error(ρ_2, exact_dense)
        err_dynamic = relative_frobenius_error(ρ_dynamic, exact_dense)

        @test err_2 < err_1
        @test err_dynamic ≤ err_1
        @test err_2 ≤ 1e-3

        energy_0 = real(tr(ρ0_dense * H_dense))
        energy_2 = energy_expectation_mpo(last(states_2), H_mpo)
        @test abs(energy_2 - energy_0) ≤ 5e-4

        exact_sx = average_observable_dense(exact_dense, x_ops)
        exact_sz = average_observable_dense(exact_dense, z_ops)
        @test abs(mean_pauli_trace_mpo(last(states_2), x_mpos) - exact_sx) ≤ 1e-3
        @test abs(mean_pauli_trace_mpo(last(states_2), z_mpos) - exact_sz) ≤ 1e-3

        for state in states_2
            metrics = dense_density_metrics(liouville_state_to_dense(state, physical_sites))
            @test abs(liouville_trace(state, trace_bra) - 1) ≤ 5e-5
            @test metrics.hermiticity ≤ 1e-8
            @test metrics.min_eig ≥ -2e-5
        end
    end

    @testset "small dissipative TFIM: TDVP tracks dense reference" begin
        T = 0.2
        dt = 0.05
        nsteps = round(Int, T / dt)

        states_2 = tdvp_trajectory(ρ0_vec, L_diss_mpo, dt, nsteps; nsite=2, maxdim=128, cutoff=1e-10)
        dynamic = dynamic_tdvp_trajectory(
            ρ0_vec,
            L_diss_mpo,
            dt,
            nsteps;
            maxdim=128,
            cutoff=1e-10,
            plateau_patience=1,
            switch_maxdim=4,
        )

        exact_dense = reshape(exp(T * L_diss_dense) * vec0, size(ρ0_dense)...)
        err_2 = relative_frobenius_error(liouville_state_to_dense(last(states_2), physical_sites), exact_dense)
        err_dynamic = relative_frobenius_error(liouville_state_to_dense(last(dynamic.states), physical_sites), exact_dense)

        @test err_2 ≤ 2e-3
        @test err_dynamic ≤ 5e-3

        for state in (last(states_2), last(dynamic.states))
            metrics = dense_density_metrics(liouville_state_to_dense(state, physical_sites))
            @test abs(liouville_trace(state, trace_bra) - 1) ≤ 1e-5
            @test metrics.hermiticity ≤ 1e-7
            @test metrics.min_eig ≥ -1e-7
        end
    end
end
