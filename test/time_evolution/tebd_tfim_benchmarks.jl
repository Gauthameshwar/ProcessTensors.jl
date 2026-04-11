using ProcessTensors
using ITensors
using LinearAlgebra
using Test

if !isdefined(Main, :dense_hamiltonian_matrix)
    include(joinpath(@__DIR__, "tebd_test_utils.jl"))
end

# Build the TFIM Hamiltonian, e.g. `-J Z_i Z_{i+1} - h X_i`.
function tfim_hamiltonian(N::Int; J::Float64=1.0, h::Float64=3.0)
    os_H = OpSum()
    for j in 1:(N - 1)
        os_H += -J, "Z", j, "Z", j + 1
    end
    for j in 1:N
        os_H += -h, "X", j
    end
    return os_H
end

# Build local decay channels, e.g. `sqrt(γ) σ_-` on every site in Liouville form `(γ, "S-", i)`.
tfim_decay_jump_ops(N::Int; γ::Float64=0.2) = [(γ, "S-", j) for j in 1:N]

# Build a dense one-site observable embedded in the full Hilbert space, e.g. `Z` on site 3.
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

# Average a local observable over all sites for a dense state or density matrix, e.g. mean `⟨Z_i⟩`.
function average_observable_dense(state, embedded_ops)
    N = length(embedded_ops)
    if state isa AbstractVector
        return real(sum(state' * (O * state) for O in embedded_ops) / N)
    end
    return real(sum(LinearAlgebra.tr(state * O) for O in embedded_ops) / N)
end

# Compute the bipartite entanglement entropy of a dense pure state, e.g. across the center bond.
function dense_bipartite_entropy(ψ_dense::AbstractVector, physical_sites, bond::Int)
    left_dim = prod(dim.(physical_sites[1:bond]))
    right_dim = prod(dim.(physical_sites[(bond + 1):end]))
    ψ_matrix = reshape(ψ_dense, left_dim, right_dim)
    singular_values = svdvals(ψ_matrix)
    probs = real.(singular_values .^ 2)
    probs ./= sum(probs)
    return -sum(p > 1e-14 ? p * log(p) : 0.0 for p in probs)
end

# Sample unitary TEBD states at requested times by evolving incrementally from one sample to the next.
function unitary_tebd_samples(ψ0, os_H::OpSum, sample_times; dt::Float64, maxdim::Int, cutoff::Float64, order::Int=2)
    states = Vector{typeof(ψ0)}(undef, length(sample_times))
    current = copy(ψ0)
    current_time = 0.0
    for (k, target_time) in enumerate(sample_times)
        Δt = target_time - current_time
        current = tebd(current, os_H, dt, Δt; maxdim=maxdim, cutoff=cutoff, order=order)
        states[k] = current
        current_time = target_time
    end
    return states
end

# Convert the dense Liouvillian null vector into a normalized steady-state density matrix.
function dense_steady_state(L_dense::AbstractMatrix{<:Number})
    spec = eigen(ComplexF64.(L_dense))
    idx = argmin(abs.(spec.values))
    d = isqrt(size(L_dense, 1))
    ρ_ss = reshape(spec.vectors[:, idx], d, d)
    ρ_ss ./= LinearAlgebra.tr(ρ_ss)
    ρ_ss = (ρ_ss + ρ_ss') / 2
    ρ_ss ./= LinearAlgebra.tr(ρ_ss)
    return ρ_ss, spec.values[idx]
end

# Sample Liouville TEBD states at requested times by evolving incrementally from one sample to the next.
function liouville_tebd_samples(ρ0, os_H::OpSum, sample_times; jump_ops, dt::Float64, maxdim::Int, cutoff::Float64, order::Int=2)
    states = Vector{typeof(ρ0)}(undef, length(sample_times))
    current = copy(ρ0)
    current_time = 0.0
    for (k, target_time) in enumerate(sample_times)
        Δt = target_time - current_time
        current = tebd(current, os_H, dt, Δt; jump_ops=jump_ops, maxdim=maxdim, cutoff=cutoff, order=order)
        states[k] = current
        current_time = target_time
    end
    return states
end

@testset "tebd.jl: unitary TFIM quench benchmark" begin
    N = 4
    bond = div(N, 2)
    physical_sites = siteinds("S=1/2", N)
    os_H = tfim_hamiltonian(N; J=1.0, h=3.0)
    H_dense = dense_hamiltonian_matrix(os_H, physical_sites)
    ψ0 = MPS(physical_sites, fill("Up", N))
    ψ0_dense = hilbert_mps_to_dense(ψ0, physical_sites)

    sample_times = collect(0.1:0.1:1.2)
    z_ops = [dense_one_site_operator("Z", physical_sites, j) for j in 1:N]

    exact_dense_states = [exp(-1im * t * H_dense) * ψ0_dense for t in sample_times]
    exact_mz = [average_observable_dense(ψ, z_ops) for ψ in exact_dense_states]
    exact_entropy = [dense_bipartite_entropy(ψ, physical_sites, bond) for ψ in exact_dense_states]

    @test minimum(exact_mz) < 0.35
    @test exact_mz[end] > minimum(exact_mz) + 0.15

    dt_choices = (0.1, 0.05, 0.025)
    mz_errors = Float64[]
    entropy_errors = Float64[]

    for dt in dt_choices
        tebd_states = unitary_tebd_samples(ψ0, os_H, sample_times; dt=dt, maxdim=128, cutoff=1e-12, order=2)
        mz_curve = [sum(real.(expect(ψ, "Z"))) / N for ψ in tebd_states]
        entropy_curve = [entropy(ψ, bond) for ψ in tebd_states]

        push!(mz_errors, maximum(abs.(mz_curve .- exact_mz)))
        push!(entropy_errors, maximum(abs.(entropy_curve .- exact_entropy)))
    end

    @test mz_errors[2] < mz_errors[1]
    @test mz_errors[3] < mz_errors[2]
    @test entropy_errors[2] < entropy_errors[1]
    @test entropy_errors[3] < entropy_errors[2]
    @test mz_errors[3] ≤ 2e-3
    @test entropy_errors[3] ≤ 2e-3
end

@testset "tebd.jl: driven-dissipative TFIM steady-state benchmark" begin
    N = 4
    physical_sites = siteinds("S=1/2", N)
    liouv_sites_shared = liouv_sites(physical_sites)
    os_H = tfim_hamiltonian(N; J=1.0, h=1.2)
    jump_ops = tfim_decay_jump_ops(N; γ=0.5)

    ψ0 = MPS(physical_sites, fill("Up", N))
    ρ0 = to_dm(ψ0)
    ρ0_vec = to_liouville(ρ0; sites=liouv_sites_shared)
    L_dense = dense_liouvillian_matrix(os_H, jump_ops, physical_sites, liouv_sites_shared)
    ρ_ss_exact, λ0 = dense_steady_state(L_dense)
    x_ops = [dense_one_site_operator("X", physical_sites, j) for j in 1:N]
    z_ops = [dense_one_site_operator("Z", physical_sites, j) for j in 1:N]
    x_mpos = single_site_pauli_mpos("X", physical_sites)
    z_mpos = single_site_pauli_mpos("Z", physical_sites)

    @test abs(λ0) ≤ 1e-9
    @test dense_density_metrics(ρ_ss_exact).hermiticity ≤ 1e-10
    @test dense_density_metrics(ρ_ss_exact).min_eig ≥ -1e-10

    sample_times = [6.0, 10.0, 14.0]
    states = liouville_tebd_samples(
        ρ0_vec,
        os_H,
        sample_times;
        jump_ops=jump_ops,
        dt=0.05,
        maxdim=128,
        cutoff=1e-12,
        order=2,
    )

    distances = Float64[]
    residuals = Float64[]
    for state in states
        ρ_dense = liouville_state_to_dense(state, physical_sites)
        push!(distances, relative_frobenius_error(ρ_dense, ρ_ss_exact))
        push!(residuals, norm(L_dense * vec(ρ_dense)) / max(norm(vec(ρ_dense)), eps(Float64)))
    end

    final_x = mean_pauli_trace_mpo(last(states), x_mpos)
    final_z = mean_pauli_trace_mpo(last(states), z_mpos)
    steady_x = average_observable_dense(ρ_ss_exact, x_ops)
    steady_z = average_observable_dense(ρ_ss_exact, z_ops)

    @test distances[end] < distances[1]
    @test residuals[end] < residuals[1]
    @test abs(final_x - steady_x) ≤ 2e-3
    @test abs(final_z - steady_z) ≤ 2e-3

    dt_choices = (0.1, 0.05, 0.025)
    dt_distances = Float64[]
    dt_residuals = Float64[]
    dt_x_errors = Float64[]
    dt_z_errors = Float64[]

    for dt in dt_choices
        state = tebd(ρ0_vec, os_H, dt, 14.0; jump_ops=jump_ops, maxdim=128, cutoff=1e-12, order=2)
        ρ_dense = liouville_state_to_dense(state, physical_sites)
        push!(dt_distances, relative_frobenius_error(ρ_dense, ρ_ss_exact))
        push!(dt_residuals, norm(L_dense * vec(ρ_dense)) / max(norm(vec(ρ_dense)), eps(Float64)))
        push!(dt_x_errors, abs(mean_pauli_trace_mpo(state, x_mpos) - steady_x))
        push!(dt_z_errors, abs(mean_pauli_trace_mpo(state, z_mpos) - steady_z))
    end

    # Smaller dt gives smaller errors
    @test dt_distances[2] < dt_distances[1]
    @test dt_distances[3] < dt_distances[2]
    @test dt_residuals[2] < dt_residuals[1]
    @test dt_residuals[3] < dt_residuals[2]
    @test dt_x_errors[3] < dt_x_errors[1]
    @test dt_z_errors[3] < dt_z_errors[1]
    
    # Reasonable errors for the smaller dt
    @test dt_distances[3] ≤ 1e-3
    @test dt_residuals[3] ≤ 2e-3
    @test dt_x_errors[3] ≤ 1e-3
    @test dt_z_errors[3] ≤ 1e-3
end
