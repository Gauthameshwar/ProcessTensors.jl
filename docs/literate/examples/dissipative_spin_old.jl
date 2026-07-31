# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors #src
# SPDX-License-Identifier: MIT #src
# #src
# File: docs/literate/examples/dissipative_spin.jl #src
# Contributor: Gauthameshwar S. #src
# #src
# Demonstrates bulk amplitude damping in a transverse-field Ising chain with #src
# Liouville-space TEBD. #src
# #src
# Run with: #src
#   julia --project=docs docs/make.jl #src

# # Bulk-dissipative spin-chain dynamics
#
# This example studies a transverse-field Ising chain subject to local spin
# loss at every site. The state is a density matrix represented as a
# Liouville-space MPS, and its Markovian time evolution is computed with TEBD.
#
# The focus is the user-facing workflow:
#
# 1. define the physical Hamiltonian as an ordinary `OpSum`,
# 2. specify local Lindblad channels through `jump_ops`,
# 3. convert the initial density matrix to Liouville space,
# 4. pass these objects directly to `tebd`.
#
# The dissipative transverse-field Ising chain is a paradigmatic model of
# correlated open-system dynamics, widely used to study relaxation,
# decoherence, and nonequilibrium steady states.

# ## Dissipative transverse-field Ising model
#
# The Hamiltonian is
#
# ```math
# H
# =
# -J\sum_{j=1}^{N-1}Z_jZ_{j+1}
# -h\sum_{j=1}^{N}X_j .
# ```
#
# Every spin is coupled to a local loss channel,
#
# ```math
# L_j=S_j^-,
# \qquad j=1,\ldots,N,
# ```
#
# so the density matrix obeys
#
# ```math
# \frac{d\rho}{dt}
# =
# -i[H,\rho]
# +
# \gamma\sum_{j=1}^{N}
# \mathcal{D}[S_j^-]\rho ,
# ```
#
# where
#
# ```math
# \mathcal{D}[L]\rho
# =
# L\rho L^\dagger
# -\frac{1}{2}\left\{L^\dagger L,\rho\right\}.
# ```
#
# Starting from ``|\mathrm{Up}\cdots\mathrm{Up}\rangle``, the local loss removes
# spin excitations and favours `Dn`. The transverse field continually rotates
# the spins, while the Ising interaction couples their response. The long-time
# state is therefore set by the competition between coherent dynamics and
# dissipation, not by Hamiltonian energy minimisation alone.

# ## Setup

using Printf
using ITensors
using ProcessTensors
using ITensors.Ops: Trotter

const N = 6
const J = 1.0
const h = 1.2
const decay_rate = 0.5

const dt = 0.1
const final_time = 4.0
const maxdim = 96
const cutoff = 1e-10

const trace_warning_tolerance = 1e-6
const trace_assertion_tolerance = 1e-2

physical_sites = siteinds("S=1/2", N)
liouville_sites = liouv_sites(physical_sites)

initial_state = MPS(physical_sites, fill("Up", N))
initial_density = to_dm(initial_state)
initial_density_liouville =
    to_liouville(initial_density; sites=liouville_sites)

# The Hamiltonian remains a physical Hilbert-space `OpSum`.

hamiltonian = let H = OpSum()
    for j in 1:(N - 1)
        H += -J, "Z", j, "Z", j + 1
    end
    for j in 1:N
        H += -h, "X", j
    end
    H
end

# A tuple `(γ, "S-", j)` adds `γ * D[S⁻ⱼ]` to the Liouvillian. The coefficient
# is the physical decay rate itself; it is not `sqrt(γ)`.

jump_operators = [
    (decay_rate, "S-", j) for j in 1:N
]

# We monitor the spatially averaged spin components
#
# ```math
# \overline X(t)=\frac{1}{N}\sum_j\langle X_j\rangle_t,
# \qquad
# \overline Z(t)=\frac{1}{N}\sum_j\langle Z_j\rangle_t .
# ```
#
# Vectorising the observable MPOs lets us evaluate them directly against the
# Liouville-space density MPS.

mean_x_operator = let observable = OpSum()
    for j in 1:N
        observable += 1 / N, "X", j
    end
    observable
end

mean_z_operator = let observable = OpSum()
    for j in 1:N
        observable += 1 / N, "Z", j
    end
    observable
end

mean_x_liouville = to_liouville(
    MPO(mean_x_operator, physical_sites);
    sites=liouville_sites,
)

mean_z_liouville = to_liouville(
    MPO(mean_z_operator, physical_sites);
    sites=liouville_sites,
)

println("Bulk-dissipative transverse-field Ising chain")
println("----------------------------------------------")
@printf("  chain length N:       %d\n", N)
@printf("  Ising coupling J:     %.3f\n", J)
@printf("  transverse field h:   %.3f\n", h)
@printf("  bulk decay rate γ:    %.3f\n", decay_rate)
@printf("  timestep dt:          %.3f\n", dt)
@printf("  final time:           %.3f\n", final_time)
@printf("  maximum bond dim:     %d\n", maxdim)
@printf("  SVD cutoff:           %.1e\n", cutoff)

@assert isapprox(real(tr(initial_density)), 1.0; atol=1e-12)

# ## Liouville-space TEBD
#
# In Liouville space, the master equation becomes
#
# ```math
# \frac{d}{dt}|\rho(t)\rangle\rangle
# =
# \mathcal{L}|\rho(t)\rangle\rangle .
# ```
#
# TEBD approximates the short-time propagator
#
# ```math
# e^{\mathcal{L}\Delta t}
# ```
#
# by a product of local Liouville-space gates. Here we use a second-order
# decomposition, `Trotter{2}()`.
#
# The important package call is deliberately kept visible: pass the vectorised
# density, the physical Hamiltonian `OpSum`, and the local jump tuples to
# `tebd`. The Liouvillian gates are constructed internally.

nsteps = round(Int, final_time / dt)
@assert isapprox(nsteps * dt, final_time; atol=100eps(Float64))

times = collect(range(0.0; step=dt, length=nsteps + 1))
density = copy(initial_density_liouville)

mean_x_values = Float64[]
mean_z_values = Float64[]
trace_errors = Float64[]
bond_dimensions = Int[]

elapsed = @elapsed begin
    for step in eachindex(times)
        trace_value = tr(to_hilbert(density))

        push!(
            mean_x_values,
            real(inner(mean_x_liouville, density) / trace_value),
        )
        push!(
            mean_z_values,
            real(inner(mean_z_liouville, density) / trace_value),
        )
        push!(trace_errors, abs(trace_value - 1))
        push!(bond_dimensions, maxlinkdim(density))

        step == length(times) && continue

        density = tebd(
            density,
            hamiltonian,
            dt,
            dt;
            jump_ops=jump_operators,
            alg=Trotter{2}(),
            maxdim=maxdim,
            cutoff=cutoff,
        )
    end
end

max_trace_error = maximum(trace_errors; init=0.0)
max_bond_dimension = maximum(bond_dimensions; init=1)

println()
println("Liouville TEBD with bulk decay")
println("-------------------------------")
@printf("  Total time taken:  %.3f s\n", elapsed)
@printf("  final ⟨X̄⟩:        %.6f\n", mean_x_values[end])
@printf("  final ⟨Z̄⟩:        %.6f\n", mean_z_values[end])
@printf("  max trace error:   %.3e\n", max_trace_error)
@printf("  max bond dim:      %d\n", max_bond_dimension)

@assert all(isfinite, mean_x_values)
@assert all(isfinite, mean_z_values)
@assert all(value -> -1 - 1e-8 <= value <= 1 + 1e-8, mean_x_values)
@assert all(value -> -1 - 1e-8 <= value <= 1 + 1e-8, mean_z_values)
@assert max_trace_error < trace_assertion_tolerance

if max_trace_error > trace_warning_tolerance
    @warn(
        "Trace drift is larger than the preferred example tolerance.",
        max_trace_error,
        trace_warning_tolerance,
    )
end

# The compact notebook above runs one representative parameter set with
# second-order TEBD. The full plotting script
# [`scripts/tebd_tfim_dissipative.jl`](https://github.com/Gauthameshwar/ProcessTensors.jl/blob/main/scripts/tebd_tfim_dissipative.jl)
# uses the same model but performs the heavier validation work: it compares
# several timesteps and Trotter orders against dense ``e^{t\mathcal L}``
# evolution for a small chain. That script generates the figures below; the
# notebook does not repeat those parameter sweeps during the documentation
# build.
#
# ![Transverse magnetisation for the dissipative TFIM compared with exact evolution](../assets/examples/tebd_tfim_dissipative_dynamics_mx.png)
#
# ![Longitudinal magnetisation for the dissipative TFIM compared with exact evolution](../assets/examples/tebd_tfim_dissipative_dynamics_mz.png)
#
# ![Density-matrix error of dissipative TEBD with respect to exact evolution](../assets/examples/tebd_tfim_dissipative_rho_error.png)
#
# !!! note "Numerical accuracy"
#     The main approximations are the finite Trotter timestep and the MPS
#     truncation controlled by `cutoff` and `maxdim`. The dense-reference error
#     figure checks convergence directly: reducing `dt` and using the
#     higher-order Trotter decomposition should systematically improve agreement
#     with exact evolution. Trace preservation provides a lightweight diagnostic
#     for the executable notebook.
#
# ## What the result shows
#
# The initial all-`Up` state has positive longitudinal magnetisation. Bulk
# `S-` loss removes that population, so ``\overline Z(t)`` falls away from its
# initial value and approaches a stationary value. At the same time, the
# transverse field creates a nonzero ``\overline X(t)`` response. Its transient
# oscillations are damped because the environment continually erases coherent
# spin motion, while the Ising coupling makes that relaxation collective rather
# than a collection of independent one-spin decays.
#
# The late-time magnetisations do not generally coincide with the ground-state
# observables of `H`: they describe a nonequilibrium stationary state selected
# by the competition between coherent rotation, spin-spin interactions, and
# local loss. In the benchmark figures, the TEBD trajectories move toward the
# dense reference as the timestep is reduced, while the error plot makes the
# expected advantage of the higher-order decomposition explicit. Together, the
# three plots show both sides of the example: physically meaningful relaxation
# dynamics and a controlled tensor-network approximation to it.
#
# !!! summary "Example takeaways"
#     - Define the spin Hamiltonian with the same physical `OpSum` used for
#       closed-system calculations.
#     - Add homogeneous amplitude damping through local `(γ, "S-", j)` tuples.
#     - Evolve the vectorised density directly with Liouville-space `tebd`.
#     - Use ``\overline X(t)`` and ``\overline Z(t)`` to separate coherent
#       response from population relaxation.
#     - Use trace drift in the compact notebook and dense-reference errors in the
#       full script to assess numerical accuracy.
