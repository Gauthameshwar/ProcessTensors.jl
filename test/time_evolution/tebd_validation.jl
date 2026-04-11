using ProcessTensors
using ITensors
using LinearAlgebra
using Test

if !isdefined(Main, :dense_hamiltonian_matrix)
    include(joinpath(@__DIR__, "tebd_test_utils.jl"))
end

# Build the spin-chain Hamiltonian used for TEBD validation.
function spin_chain_hamiltonian(N::Int)
    os_H = OpSum()
    for j in 1:(N - 1)
        os_H += 0.5, "S+", j, "S-", j + 1
        os_H += 0.5, "S-", j, "S+", j + 1
        os_H += 1.0, "Sz", j, "Sz", j + 1
    end
    return os_H
end

# Build local amplitude-damping jumps on every site, e.g. `(0.1, "S-", 3)`.
spin_chain_jump_ops(N::Int; γ::Float64=0.1) = [(γ, "S-", j) for j in 1:N]

@testset "tebd.jl: Liouville TEBD validation against dense reference" begin
    N = 4
    physical_sites = siteinds("S=1/2", N)
    liouv_sites_shared = liouv_sites(physical_sites)
    os_H = spin_chain_hamiltonian(N)
    jump_ops = spin_chain_jump_ops(N; γ=0.1)

    ψ0 = MPS(physical_sites, fill("Up", N))
    ρ0 = to_dm(ψ0)
    ρ0_vec = to_liouville(ρ0; sites=liouv_sites_shared)
    trace_bra = vectorized_identity_state(physical_sites, liouv_sites_shared)
    ρ0_dense = hilbert_mpo_to_dense(ρ0, physical_sites)
    L_dense = dense_liouvillian_matrix(os_H, jump_ops, physical_sites, liouv_sites_shared)

    @testset "trace, Hermiticity, and positivity along trajectory" begin
        dt = 0.05
        nsteps = 4
        trace_tol = 3e-7
        states = tebd_trajectory(
            ρ0_vec,
            os_H,
            dt,
            nsteps;
            jump_ops=jump_ops,
            maxdim=128,
            cutoff=1e-12,
            order=2,
        )

        for (step, state) in enumerate(states)
            ρ_dense = liouville_state_to_dense(state, physical_sites)
            metrics = dense_density_metrics(ρ_dense)
            trace_from_bra = inner(trace_bra, state)

            @test abs(trace_from_bra - 1) ≤ trace_tol
            @test abs(metrics.trace - 1) ≤ trace_tol
            @test metrics.hermiticity ≤ 1e-8
            @test metrics.min_eig ≥ -1e-8

            if step > 1
                @test maxlinkdim(state) ≤ 128
            end
        end
    end

    @testset "dense exp(tL) agreement improves with timestep refinement" begin
        T = 0.2
        exact_dense = reshape(exp(T * L_dense) * vec(ρ0_dense), size(ρ0_dense)...)
        dts = (0.1, 0.05, 0.025)
        errors = Float64[]

        for dt in dts
            state = tebd(ρ0_vec, os_H, dt, T; jump_ops=jump_ops, maxdim=128, cutoff=1e-12, order=2)
            ρ_tebd_dense = liouville_state_to_dense(state, physical_sites)
            push!(errors, relative_frobenius_error(ρ_tebd_dense, exact_dense))
        end

        @test errors[2] < errors[1]
        @test errors[3] < errors[2]
        @test errors[3] ≤ 5e-4
    end

    @testset "reference and moderate truncation stay close" begin
        T = 0.2
        ρ_tight = tebd(ρ0_vec, os_H, 0.025, T; jump_ops=jump_ops, maxdim=128, cutoff=1e-12, order=2)
        ρ_loose = tebd(ρ0_vec, os_H, 0.025, T; jump_ops=jump_ops, maxdim=32, cutoff=1e-9, order=2)

        err = relative_frobenius_error(
            liouville_state_to_dense(ρ_loose, physical_sites),
            liouville_state_to_dense(ρ_tight, physical_sites),
        )

        @test err ≤ 1e-6
    end
end
