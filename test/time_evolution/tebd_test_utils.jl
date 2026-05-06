using ProcessTensors
using ITensors
using LinearAlgebra

# Turn a dense matrix into a Hilbert MPO on the supplied physical sites.
function hilbert_matrix_to_mpo(M::AbstractMatrix{<:Number}, physical_sites)
    dims = vcat(dim.(prime.(physical_sites)), dim.(physical_sites))
    T = ITensor(reshape(ComplexF64.(M), Tuple(dims)), prime.(physical_sites)..., physical_sites...)
    return ProcessTensors.MPO(T, physical_sites)
end

# Contract all tensors in a network core into one ITensor, e.g. an MPS or MPO into its dense coefficient tensor.
function contract_core(core)
    T = core[1]
    for j in 2:length(core)
        T *= core[j]
    end
    return T
end

# Convert a Hilbert MPO into a dense matrix, e.g. for `N=4` spins this returns a `16×16` matrix.
function hilbert_mpo_to_dense(ρ::AbstractMPO{Hilbert}, physical_sites)
    T = contract_core(ρ.core)
    A = Array(T, prime.(physical_sites)..., physical_sites...)
    return reshape(ComplexF64.(A), prod(dim.(physical_sites)), prod(dim.(physical_sites)))
end

# Convert a Hilbert MPS into a dense state vector, e.g. for `N=4` spins this returns length `16`.
function hilbert_mps_to_dense(ψ::AbstractMPS{Hilbert}, physical_sites)
    T = contract_core(ψ.core)
    return vec(ComplexF64.(Array(T, physical_sites...)))
end

# Convert a Liouville MPS into the dense density matrix it represents, e.g. `ρ_vec -> ρ`.
liouville_state_to_dense(ρ_vec::AbstractMPS{Liouville}, physical_sites) =
    hilbert_mpo_to_dense(to_hilbert(ρ_vec), physical_sites)

"""One-site operator as `MPO{Hilbert}` on `physical_sites`, one rank-1 MPO per site `j`."""
function single_site_pauli_mpos(op::AbstractString, physical_sites)
    N = length(physical_sites)
    return MPO{Hilbert}[
        let os = OpSum()
            os += 1.0, op, j
            MPO(os, physical_sites)
        end for j in 1:N
    ]
end

"""Mean ``(1/N) \\sum_j \\mathrm{Tr}(\\rho O_j)`` for ``\\rho`` given as `MPS{Liouville}`."""
function mean_operator_trace_mpo(ρ_vec::MPS{Liouville}, operator_mpos::Vector{MPO{Hilbert}})
    ρ_h = to_hilbert(ρ_vec)
    Ns = length(operator_mpos)
    s = 0.0
    for O in operator_mpos
        ρO = apply(O, ρ_h; alg="naive", truncate=false)
        s += real(tr(ρO))
    end
    return s / Ns
end

"""Mean ``(1/N) \\sum_j \\mathrm{Tr}(\\rho O_j)`` for ``\\rho`` given as `MPS{Liouville}` (exact MPO apply + trace)."""
mean_pauli_trace_mpo(ρ_vec::MPS{Liouville}, pauli_mpos::Vector{MPO{Hilbert}}) =
    mean_operator_trace_mpo(ρ_vec, pauli_mpos)

"""Expectation ``Tr(ρ O)`` for Liouville state `ρ_vec` and Hilbert-space MPO `O`."""
function expectation_trace_mpo(ρ_vec::MPS{Liouville}, O::MPO{Hilbert})
    ρ_h = to_hilbert(ρ_vec)
    ρO = apply(O, ρ_h; alg="naive", truncate=false)
    return tr(ρO)
end

# Build the vectorized identity used for the bra-trace check, e.g. `⟨⟨I|ρ⟩⟩ = tr(ρ)`.
function vectorized_identity_state(physical_sites, liouv_sites_shared)
    d = prod(dim.(physical_sites))
    identity_mpo = hilbert_matrix_to_mpo(Matrix{ComplexF64}(I, d, d), physical_sites)
    return to_liouville(identity_mpo; sites=liouv_sites_shared)
end

liouville_trace(ρ_vec::AbstractMPS{Liouville}, trace_bra::AbstractMPS{Liouville}) = inner(trace_bra, ρ_vec)

function hermiticity_defect_mpo(ρ_vec::AbstractMPS{Liouville})
    ρ_h = to_hilbert(ρ_vec)
    return norm(ρ_h - dag(ρ_h)) / max(norm(ρ_h), eps(Float64))
end

function energy_expectation_mpo(ρ_vec::MPS{Liouville}, H_mpo::MPO{Hilbert})
    return real(expectation_trace_mpo(ρ_vec, H_mpo))
end

# Build a dense Hamiltonian matrix from an OpSum, e.g. for ED against Hilbert-space `tebd`.
dense_hamiltonian_matrix(os_H::OpSum, physical_sites) =
    hilbert_mpo_to_dense(MPO(os_H, physical_sites), physical_sites)

# Build one dense Liouvillian matrix by acting on every basis operator `E_ab`.
function dense_liouvillian_matrix(os_H::OpSum, jump_ops, physical_sites, liouv_sites_shared)
    L_mpo = MPO_Liouville(os_H, liouv_sites_shared; jump_ops=jump_ops)
    d = prod(dim.(physical_sites))
    d2 = d * d

    L_dense = zeros(ComplexF64, d2, d2)
    for b in 1:d
        for a in 1:d
            q = a + (b - 1) * d
            E = zeros(ComplexF64, d, d)
            E[a, b] = 1.0
            basis_q = to_liouville(hilbert_matrix_to_mpo(E, physical_sites); sites=liouv_sites_shared)
            σ_q = apply(L_mpo, basis_q; cutoff=0.0, maxdim=typemax(Int))
            L_dense[:, q] = vec(liouville_state_to_dense(σ_q, physical_sites))
        end
    end

    return L_dense
end

# Evolve repeatedly by one TEBD step so invariants can be checked at every sampled time.
function tebd_trajectory(ρ0, os_H, dt::Real, nsteps::Integer; jump_ops, maxdim::Int, cutoff::Float64, order::Int=2)
    states = Vector{typeof(ρ0)}(undef, nsteps + 1)
    states[1] = copy(ρ0)
    current = copy(ρ0)
    for step in 1:nsteps
        current = tebd(current, os_H, dt, dt; jump_ops=jump_ops, maxdim=maxdim, cutoff=cutoff, order=order)
        states[step + 1] = current
    end
    return states
end

function tdvp_trajectory(
    ρ0,
    operator,
    dt::Real,
    nsteps::Integer;
    nsite::Int,
    maxdim::Int,
    cutoff::Float64,
)
    states = Vector{typeof(ρ0)}(undef, nsteps + 1)
    states[1] = copy(ρ0)
    current = copy(ρ0)
    for step in 1:nsteps
        current = tdvp(operator, dt, current; time_step=dt, nsite=nsite, maxdim=maxdim, cutoff=cutoff, outputlevel=0)
        states[step + 1] = current
    end
    return states
end

function dynamic_tdvp_trajectory(
    ρ0,
    operator,
    dt::Real,
    nsteps::Integer;
    maxdim::Int,
    cutoff::Float64,
    plateau_patience::Int=2,
    switch_maxdim::Union{Nothing, Int}=nothing,
)
    states = Vector{typeof(ρ0)}(undef, nsteps + 1)
    nsites = Vector{Int}(undef, nsteps)
    bond_dims = Vector{Int}(undef, nsteps + 1)
    states[1] = copy(ρ0)
    bond_dims[1] = maxlinkdim(ρ0)
    current = copy(ρ0)
    current_nsite = 2
    stagnant_steps = 0

    for step in 1:nsteps
        nsites[step] = current_nsite
        prev_bond_dim = maxlinkdim(current)
        current = tdvp(
            operator,
            dt,
            current;
            time_step=dt,
            nsite=current_nsite,
            maxdim=maxdim,
            cutoff=cutoff,
            outputlevel=0,
        )
        states[step + 1] = current
        bond_dims[step + 1] = maxlinkdim(current)

        if current_nsite == 2
            if bond_dims[step + 1] <= prev_bond_dim
                stagnant_steps += 1
            else
                stagnant_steps = 0
            end

            reached_switch_cap = !isnothing(switch_maxdim) && bond_dims[step + 1] >= switch_maxdim
            if stagnant_steps >= plateau_patience || reached_switch_cap
                current_nsite = 1
            end
        end
    end

    return (; states, nsites, bond_dims)
end

# Summarize dense diagnostics in one place, e.g. trace, Hermiticity defect, and minimum eigenvalue.
function dense_density_metrics(ρ_dense::AbstractMatrix{<:Number})
    ρ = ComplexF64.(ρ_dense)
    herm_defect = norm(ρ - ρ') / max(norm(ρ), eps(Float64))
    ρ_herm = (ρ + ρ') / 2
    λmin = minimum(real.(eigvals(Hermitian(ρ_herm))))
    return (trace=tr(ρ), hermiticity=herm_defect, min_eig=λmin)
end

# Compare two dense density matrices in relative Frobenius norm, e.g. TEBD vs exact `exp(tL)`.
relative_frobenius_error(A::AbstractMatrix{<:Number}, B::AbstractMatrix{<:Number}) =
    norm(ComplexF64.(A) - ComplexF64.(B)) / max(norm(ComplexF64.(B)), eps(Float64))
