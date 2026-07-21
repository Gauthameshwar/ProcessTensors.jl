# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors #src
# SPDX-License-Identifier: MIT #src
# #src
# File: docs/literate/examples/tebd_time_evolution.jl #src
# Contributor: Gauthameshwar S. #src
# #src
# Demonstrates unitary TEBD for the same spin chain in Hilbert and Liouville #src
# space, with a compact exact small-system reference. #src

# # TEBD time evolution
#
# Time-evolving block decimation (TEBD) approximates the propagator by a
# sequence of local Suzuki–Trotter gates. This example keeps the main comparison
# visible: evolve a pure-state MPS in Hilbert space, then evolve its vectorized
# density matrix in Liouville space.
#
# Both representations describe the same closed-system physics. This page keeps
# a compact Hilbert/Liouville comparison. For timestep sweeps, Trotter-order
# benchmarks, detailed error diagnostics, and plotting, see the full script
# [`scripts/tebd_tfim_unitary.jl`](https://github.com/Gauthameshwar/ProcessTensors.jl/blob/main/scripts/tebd_tfim_unitary.jl).

# ## Transverse-field Ising model
#
# We study
#
# ```math
# H=-J\sum_{j=1}^{N-1}Z_jZ_{j+1}-h\sum_{j=1}^{N}X_j
# ```
#
# from the product state
# ``|\psi(0)\rangle=|\mathrm{Up}\cdots\mathrm{Up}\rangle``.

using ITensors
using LinearAlgebra
using ProcessTensors

const N = 4
const J = 1.0
const h = 1.2
const dt = 0.1
const final_time = 2.0
const nsteps = round(Int, final_time / dt)
const maxdim = 64
const cutoff = 1e-12

sites = siteinds("S=1/2", N)
liouville_sites = liouv_sites(sites)

H = let H_local = OpSum()
    for j in 1:(N - 1)
        H_local += -J, "Z", j, "Z", j + 1
    end
    for j in 1:N
        H_local += -h, "X", j
    end
    H_local
end

H_mpo = MPO(H, sites)
initial_state = MPS(sites, fill("Up", N))
initial_density = to_dm(initial_state)
initial_density_liouville =
    to_liouville(initial_density; sites=liouville_sites)

mean_x = let observable = OpSum()
    for j in 1:N
        observable += 1 / N, "X", j
    end
    observable
end
mean_x_mpo = MPO(mean_x, sites)
mean_x_liouville = to_liouville(mean_x_mpo; sites=liouville_sites)

@assert isapprox(real(inner(initial_state, initial_state)), 1; atol=1e-12)

# ## Exact small-system reference
#
# Exact diagonalization is practical here only because ``N=4``. The tensor
# contractions below expose the dense Hamiltonian, initial state, and mean-spin
# observable directly. These will later be compared to the TEBD results we obtain 
# using our `tebd` function.

dimension = prod(dim.(sites))

H_tensor = foldl(*, H_mpo)
H_dense = reshape(
    ComplexF64.(Array(H_tensor, prime.(sites)..., sites...)),
    dimension,
    dimension,
)

state_tensor = foldl(*, initial_state)
state_dense = vec(ComplexF64.(Array(state_tensor, sites...)))
density_dense = state_dense * state_dense'

mean_x_tensor = foldl(*, mean_x_mpo)
mean_x_dense = reshape(
    ComplexF64.(Array(mean_x_tensor, prime.(sites)..., sites...)),
    dimension,
    dimension,
)

function exact_density_at(
    time::Real,
    H_dense::AbstractMatrix,
    density0::AbstractMatrix,
)
    propagator = exp(-1im * time * H_dense)
    return propagator * density0 * propagator'
end

function exact_sx_trajectory(
    H_dense::AbstractMatrix,
    density0::AbstractMatrix,
    mean_x_dense::AbstractMatrix,
    times::AbstractVector,
)
    return [
        real(tr(exact_density_at(time, H_dense, density0) * mean_x_dense))
        for time in times
    ]
end

times = collect(range(0.0; step=dt, length=nsteps + 1))
sx_exact =
    exact_sx_trajectory(H_dense, density_dense, mean_x_dense, times)

# ## Hilbert-space TEBD
#
# In Hilbert space, `tebd` applies the Hamiltonian gates directly to
# ``|\psi\rangle``. We use second-order Trotter splitting and record
# ``\langle\bar X\rangle`` after every step.

hilbert_trajectory = let
    state = copy(initial_state)
    sx = Float64[real(inner(state', mean_x_mpo, state))]
    elapsed = 0.0
    for _ in 1:nsteps
        elapsed += @elapsed state = tebd(
            state,
            H,
            dt,
            dt;
            alg=Trotter{2}(),
            maxdim=maxdim,
            cutoff=cutoff,
        )
        push!(sx, real(inner(state', mean_x_mpo, state)))
    end

    (; state, sx, elapsed)
end

hilbert_density = to_dm(hilbert_trajectory.state)
hilbert_density_tensor = foldl(*, hilbert_density)
hilbert_density_dense = reshape(
    ComplexF64.(
        Array(hilbert_density_tensor, prime.(sites)..., sites...)
    ),
    dimension,
    dimension,
)

exact_final_density =
    exact_density_at(final_time, H_dense, density_dense)
hilbert_error =
    norm(hilbert_density_dense - exact_final_density) /
    norm(exact_final_density)

println("TEBD(2) on Hilbert space with dt=$(dt)")
println("    Wall time to final state: $(round(hilbert_trajectory.elapsed, digits=4)) s")
println("    Norm error of the final matrix: $(round(hilbert_error, digits=6))")
println("    Max bond dim of final state: $(maxlinkdim(hilbert_trajectory.state))")

@assert all(isfinite, hilbert_trajectory.sx)
@assert hilbert_error < 0.05

# After sweeping timesteps and Trotter orders in
# [`scripts/tebd_tfim_unitary.jl`](https://github.com/Gauthameshwar/ProcessTensors.jl/blob/main/scripts/tebd_tfim_unitary.jl),
# the Hilbert-space dynamics and density-matrix error against ED look like:
#
# ![Hilbert-space TEBD and exact mean spin](../assets/examples/tebd_tfim_unitary_hilbert_dynamics_mx.png)
# ![Hilbert-space TEBD error with respect to ED](../assets/examples/tebd_tfim_unitary_hilbert_rho_error.png)
# ## Liouville-space TEBD
#
# In Liouville space, the state is ``|\rho\rangle\rangle`` and the same public
# `tebd` function constructs the commutator Liouvillian internally. No jump
# operators are supplied, so this remains closed unitary dynamics.

empty_jumps = Tuple{Number,String,Int}[]

liouville_trajectory = let
    density = copy(initial_density_liouville)
    sx = Float64[real(inner(mean_x_liouville, density))]
    elapsed = 0.0
    for _ in 1:nsteps
        elapsed += @elapsed density = tebd(
            density,
            H,
            dt,
            dt;
            jump_ops=empty_jumps,
            alg=Trotter{2}(),
            maxdim=maxdim,
            cutoff=cutoff,
        )
        push!(sx, real(inner(mean_x_liouville, density)))
    end

    (; density, sx, elapsed)
end

liouville_density = to_hilbert(liouville_trajectory.density)
liouville_density_tensor = foldl(*, liouville_density)
liouville_density_dense = reshape(
    ComplexF64.(
        Array(liouville_density_tensor, prime.(sites)..., sites...)
    ),
    dimension,
    dimension,
)

liouville_error =
    norm(liouville_density_dense - exact_final_density) /
    norm(exact_final_density)

println("TEBD(2) on Liouville space with dt=$(dt)")
println("    Wall time to final state: $(round(liouville_trajectory.elapsed, digits=4)) s")
println("    Norm error of the final matrix: $(round(liouville_error, digits=6))")
println("    Max bond dim of final state: $(maxlinkdim(liouville_trajectory.density))")

@assert all(isfinite, liouville_trajectory.sx)
@assert maximum(abs.(hilbert_trajectory.sx - sx_exact)) < 0.05
@assert maximum(abs.(liouville_trajectory.sx - sx_exact)) < 0.05
@assert liouville_error < 0.05

# The Liouville trajectory is consistent with Hilbert TEBD. The final bond
# dimension is typically larger in Liouville space because one evolves a
# vectorized density matrix rather than a pure-state MPS.
#
# The corresponding figures from
# [`scripts/tebd_tfim_unitary.jl`](https://github.com/Gauthameshwar/ProcessTensors.jl/blob/main/scripts/tebd_tfim_unitary.jl)
# are:
#
# ![Liouville-space TEBD and exact mean spin](../assets/examples/tebd_tfim_unitary_liouville_dynamics_mx.png)
# ![Liouville-space TEBD error with respect to ED](../assets/examples/tebd_tfim_unitary_liouville_rho_error.png)
#
# !!! summary "Example takeaways"
#     - Hilbert TEBD evolves `MPS{Hilbert}` directly under the Hamiltonian.
#     - Liouville TEBD evolves `MPS{Liouville}` under the corresponding
#       commutator, with the same `tebd` entry point.
#     - Liouville evolution generally needs a larger bond dimension than the
#       matching pure-state Hilbert run.
#     - Exact diagonalization is a small-system check; TEBD is the scalable
#       tensor-network calculation.
#     - For the full implementation (timestep sweeps, Trotter-order comparisons,
#       detailed errors, live output, and plotting), run
#       `scripts/tebd_tfim_unitary.jl`.
