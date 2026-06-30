# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors #src
# SPDX-License-Identifier: MIT #src
# #src
# File: docs/literate/tutorials/04_dissipative_dynamics.jl #src
# Contributor: Gauthameshwar S. #src
# #src
# Literate tutorial source: dissipative open dynamics in Liouville space. #src

# # Dissipative Dynamics
#

# In the unitary-dynamics tutorial, Liouville space was a second route to the
# same closed-system physics. For open systems, that picture becomes primary:
# the state is generally a mixed **density matrix** `ρ`, not a pure wavefunction,
# and the dynamics is usually written as a master equation rather than a
# Schrödinger equation.
#
# After vectorisation,
#
# ```math
# \rho \longmapsto |\rho\rangle\rangle,
# ```
#
# a Liouvillian generator becomes an ordinary linear operator on the
# vectorised density,
#
# ```math
# |\rho\rangle\rangle \mapsto \mathcal{L}|\rho\rangle\rangle.
# ```
#
# The familiar MPS/MPO workflow then carries over with different types and
# indices. Here is a quick summary table of how the Hilbert and Liouville
# space handles the same ideas in different ways.
#
# ```@raw html
# <table>
# <thead>
# <tr><th>Concept</th><th>Hilbert space</th><th>Liouville space</th></tr>
# </thead>
# <tbody>
# <tr><td>Physical object</td><td>pure or mixed ket $|\psi\rangle$ or density $\rho$</td><td>vectorised density $|\rho\rangle\rangle$</td></tr>
# <tr><td>Operator / generator</td><td>Hamiltonian $H$</td><td>Liouvillian superoperator $\mathcal{L}$</td></tr>
# <tr><td>Site indices</td><td>physical <code>sites</code></td><td><code>liouv_sites(sites)</code> (local dim $d^2$)</td></tr>
# <tr><td>Closed-system dynamics</td><td>$\mathrm{i}\hbar\,\partial_t |\psi\rangle = H|\psi\rangle$</td><td>$\mathrm{i}\hbar\,\partial_t |\rho\rangle\rangle = [H,\cdot]|\rho\rangle\rangle$</td></tr>
# <tr><td>Open-system dynamics</td><td>not possible with a single-state Schrödinger equation</td><td>local master equation for $\rho$, linear in $|\rho\rangle\rangle$</td></tr>
# <tr><td>One-step propagator (TEBD)</td><td>$e^{-iH\Delta t}$</td><td>$e^{\mathcal{L}\Delta t}$</td></tr>
# <tr><td>Package evolution calls</td><td><code>tebd(ψ, H, …)</code> / <code>tdvp(H_mpo, …)</code></td><td><code>tebd(ρL, H, …; jump_ops=…)</code> / <code>tdvp(L_mpo, …)</code></td></tr>
# </tbody>
# </table>
# ```
#
# The tutorial below demonstrates the local Liouvillian action, 
# and time evolves with Liouville TEBD and TDVP on a single-spin and two-spin model.
#
# ## Setup
#
# The helper functions below are for small dense checks in this tutorial.

using ITensors
import ITensorMPS
import LinearAlgebra
using ProcessTensors

roundreal(x; digits=8) = round(real(x); digits=digits)

function dense_mpo_matrix(ρL::AbstractMPS{Liouville}, sites)
    W = to_hilbert(ρL)
    T = foldl(*, W)

    A = Array(T, prime.(sites)..., sites...)
    D = prod(dim.(sites))

    return reshape(ComplexF64.(A), D, D)
end

function density_matrix_properties(ρ::AbstractMatrix)
    ρc = ComplexF64.(ρ)
    ρh = (ρc + ρc') / 2

    trace = LinearAlgebra.tr(ρc)
    hermiticity = LinearAlgebra.norm(ρc - ρc') / max(LinearAlgebra.norm(ρc), eps(Float64))
    min_eig = minimum(real.(LinearAlgebra.eigvals(LinearAlgebra.Hermitian(ρh))))

    return (; trace, hermiticity, min_eig)
end

# ## [A single spin with amplitude damping](@id dissipative-lindblad-mpo)
#
# We begin with one spin-1/2 system. The Hamiltonian is
#
# ```math
# H =
# \frac{\omega}{2}S^z.
# ```
#
# The dissipative jump is
#
# ```math
# L = S^-.
# ```
#
# This describes amplitude damping: population flows from the upper spin state
# into the lower spin state.
#
# We choose the initial state
#
# ```math
# |+\rangle =
# \frac{1}{\sqrt{2}}
# \left(
# |\uparrow\rangle
# +
# |\downarrow\rangle
# \right).
# ```
#
# This state has both population and coherence, so damping has something visible
# to act on.
#
# ### Dissipative jump operators
#
# The full Markovian generator has Hamiltonian and dissipative parts,
#
# ```math
# \frac{d\rho}{dt}
# =
# -i[H,\rho]
# +
# \sum_\mu
# \gamma_\mu
# \left(
# L_\mu \rho L_\mu^\dagger
# -
# \frac{1}{2}
# \{L_\mu^\dagger L_\mu,\rho\}
# \right).
# ```
#
# In `ProcessTensors.jl`, local jumps are passed as tuples:
#
# ```julia
# jump_ops = [(γ, "S-", 1)]
# ```
#
# This means: add the dissipator $S^-$ with jump rate $\gamma$ at site 1
# such that the local Liouvillian action of this jump is given by
#
# ```math
# \mathcal{D}[S^-](\rho) =
# S^-\rho S^+
# -
# \frac{1}{2}
# \{S^+ S^-,\rho\}.
# ```
#
# For this single-spin model the Liouville generator is
#
# ```math
# \mathcal{L}
# =
# -i[H,\cdot]
# +
# \gamma\mathcal{D}[S^-](\cdot).
# ```

# !!! warning "Rate convention"
#     The tuple `(γ, "S-", 1)` stores the jump rate `γ`. The package inserts
#     `γ * D[S-](ρ)`. Do not pass `sqrt(γ)` in this tuple unless you deliberately
#     want the operator coefficient itself to contain a square root.

ω = 1.3
γ = 0.4

sites = siteinds("S=1/2", 1)
sites_L = liouv_sites(sites)

ψ0 = MPS(ComplexF64[1 / sqrt(2), 1 / sqrt(2)], sites)
ρ0 = to_dm(ψ0)
ρL0 = to_liouville(ρ0; sites=sites_L)

println("Initial Liouville state:")
println(ρL0)

@assert ψ0 isa MPS{Hilbert}
@assert ρ0 isa MPO{Hilbert}
@assert ρL0 isa MPS{Liouville}

# The Hamiltonian is still written as an ordinary Hilbert-space `OpSum`.
# Dissipation enters through `jump_ops`, as introduced above.
# To see how the `OpSum` terms construct a dissipative $\mathcal{L}$, see the
# [Liouville superoperators and OpSums](@ref liouville-superoperators-and-opsums)
# section of the Liouville space theory page.

H = OpSum()
H += (ω / 2), "Sz", 1

jump_ops = [(γ, "S-", 1)]

# Build the Liouville-space generator MPO.

L_mpo = MPO_Liouville(H, sites_L; jump_ops=jump_ops)

println("Liouvillian MPO:")
println(L_mpo)

@assert L_mpo isa MPO{Liouville}

# !!! tip "Primary package pattern"
#     The key construction is:
#
#     ```julia
#     L_mpo = MPO_Liouville(H, sites_L; jump_ops=jump_ops)
#     ```
#
#     This is the dissipative analogue of building a Hamiltonian MPO.

# ### Checking the local Liouvillian action
#
# For one spin, the vectorised density matrix has four components:
#
# ```math
# |\rho\rangle\rangle
# =
# \begin{pmatrix}
# \rho_{00}\\
# \rho_{10}\\
# \rho_{01}\\
# \rho_{11}
# \end{pmatrix}.
# ```
#
# For the model above, the analytical action is
#
# ```math
# \mathcal{L}
# \begin{pmatrix}
# \rho_{00}\\
# \rho_{10}\\
# \rho_{01}\\
# \rho_{11}
# \end{pmatrix}
# =
# \begin{pmatrix}
# -\gamma\rho_{00}\\
# (i\omega/2-\gamma/2)\rho_{10}\\
# (-i\omega/2-\gamma/2)\rho_{01}\\
# \gamma\rho_{00}
# \end{pmatrix}.
# ```
#
# Let us check that the package MPO produces exactly this local action.

sL = only(sites_L)

ρvec0 = ComplexF64.(Array(ρL0[1], sL))
ρ00, ρ10, ρ01, ρ11 = ρvec0

dρ_expected = ComplexF64[
    -γ * ρ00,
    (1im * ω / 2 - γ / 2) * ρ10,
    (-1im * ω / 2 - γ / 2) * ρ01,
    γ * ρ00,
]

dρ_from_mpo = ComplexF64.(Array(L_mpo[1] * ρL0[1], prime(sL)))

println("‖L|ρ⟩⟩ - analytical formula‖ = ",
        LinearAlgebra.norm(dρ_from_mpo - dρ_expected))

@assert LinearAlgebra.norm(dρ_from_mpo - dρ_expected) < 1e-10

# !!! note "Why this check matters"
#     `MPO_Liouville` is not just wrapping a dense matrix. It builds the local
#     superoperator structure from the Hamiltonian and the jump terms. The
#     single-spin formula lets us see that structure explicitly.

# ## [Evolving the density matrix](@id dissipative-evolving-density-matrix)
#
# We now time evolve the density matrix.
#
# In Hilbert space, unitary TEBD used gates for
#
# ```math
# e^{-iH\Delta t}.
# ```
#
# In Liouville space, dissipative TEBD uses gates for
#
# ```math
# e^{\mathcal{L}\Delta t}.
# ```
#
# The package call is deliberately simple:
#
# ```julia
# ρL_t = tebd(ρL0, H, dt, T; jump_ops=jump_ops)
# ```
#
# Notice that we pass the Hilbert-space Hamiltonian `H` and the dissipative jumps.
# The Liouville generator is built internally.
#
# On this one-site validation example we use `Trotter{4}()` so the gate
# approximation stays close to the dense `exp(TL)` reference below.

dt = 0.02
T = 0.2

ρL_tebd = tebd(
    ρL0,
    H,
    dt,
    T;
    jump_ops=jump_ops,
    alg=Trotter{4}(),
    maxdim=16,
    cutoff=1e-12,
)

ρ_tebd = dense_mpo_matrix(ρL_tebd, sites)

println("Density matrix after Liouville TEBD:")
println(roundreal.(ρ_tebd))

metrics_tebd = density_matrix_properties(ρ_tebd)

println("Trace after TEBD:       ", real(metrics_tebd.trace))
println("Hermiticity defect:     ", metrics_tebd.hermiticity)
println("Minimum eigenvalue:     ", metrics_tebd.min_eig)

@assert ρL_tebd isa MPS{Liouville}
@assert abs(real(metrics_tebd.trace) - 1) < 1e-6
@assert metrics_tebd.hermiticity < 1e-6
@assert metrics_tebd.min_eig > -1e-6

# !!! info "Physical sanity checks"
#     A density matrix should remain trace-one, Hermitian, and positive
#     semidefinite. Small violations usually indicate numerical errors from
#     truncation, time stepping, or an inconsistent Liouville-index convention.

# ### Exact dense check for the tiny system
#
# Since this is only one spin, we can also build the dense Liouvillian matrix
# and compare against
#
# ```math
# |\rho(T)\rangle\rangle_{\mathrm{exact}}
# =
# e^{T\mathcal{L}}
# |\rho(0)\rangle\rangle.
# ```
#
# This is a validation check, not the scalable algorithm.

L_dense = Matrix(Array(L_mpo[1], prime(sL), sL))

ρvec_exact = LinearAlgebra.exp(T * L_dense) * ρvec0
ρ_exact = reshape(ρvec_exact, 2, 2)

err_tebd = LinearAlgebra.norm(ρ_tebd - ρ_exact) /
           max(LinearAlgebra.norm(ρ_exact), eps(Float64))

println("Relative TEBD error against dense exp(TL): ", err_tebd)

@assert err_tebd < 1e-8

# ## Scaling the dissipative dynamics
#
# The single-spin example was chosen because it has a clean analytical check.
# The package interface, however, is already many-body.
#
# `ITensorMPS.tdvp` currently requires at least two sites, so we demonstrate
# both TEBD and TDVP on a two-spin model:
#
# ```math
# H =
# J S^z_1S^z_2
# +
# h(S^x_1+S^x_2),
# ```
#
# with local amplitude damping on both sites:
#
# ```math
# L_1 = S^-_1,
# \qquad
# L_2 = S^-_2.
# ```

chain_sites = siteinds("S=1/2", 2)
chain_sites_L = liouv_sites(chain_sites)

ψ_chain0 = MPS(chain_sites, ["Up", "Dn"])
ρ_chain0 = to_dm(ψ_chain0)
ρL_chain0 = to_liouville(ρ_chain0; sites=chain_sites_L)

H_chain = OpSum()
H_chain += 1.0, "Sz", 1, "Sz", 2
H_chain += 0.4, "Sx", 1
H_chain += 0.4, "Sx", 2

chain_jumps = [(0.15, "S-", 1), (0.15, "S-", 2)]

L_chain = MPO_Liouville(H_chain, chain_sites_L; jump_ops=chain_jumps)

chain_dt = 0.025
chain_T = 0.1

ρL_chain_t = tebd(
    ρL_chain0,
    H_chain,
    chain_dt,
    chain_T;
    jump_ops=chain_jumps,
    alg=Trotter{2}(),
    maxdim=32,
    cutoff=1e-12,
)

ρ_chain_t = dense_mpo_matrix(ρL_chain_t, chain_sites)
chain_metrics = density_matrix_properties(ρ_chain_t)

println("Two-spin dissipative TEBD:")
println("  trace              = ", real(chain_metrics.trace))
println("  hermiticity defect = ", chain_metrics.hermiticity)
println("  min eigenvalue     = ", chain_metrics.min_eig)
println("  max bond dimension = ", maxlinkdim(ρL_chain_t))

@assert abs(real(chain_metrics.trace) - 1) < 1e-6
@assert chain_metrics.hermiticity < 1e-6
@assert chain_metrics.min_eig > -1e-6
@assert maxlinkdim(ρL_chain_t) ≤ 32

# ### TDVP uses the same Liouville MPO
#
# TEBD is not the only option. We can also evolve the same vectorised density
# matrix with TDVP on the two-site chain:
#
# ```julia
# ρL_tdvp = tdvp(L_chain, chain_T, ρL_chain0; time_step=chain_dt)
# ```

ρL_chain_tdvp = tdvp(
    L_chain,
    chain_T,
    ρL_chain0;
    time_step=chain_dt,
    nsite=1,
    maxdim=32,
    cutoff=1e-12,
    outputlevel=0,
)

ρ_chain_tdvp = dense_mpo_matrix(ρL_chain_tdvp, chain_sites)
chain_tdvp_metrics = density_matrix_properties(ρ_chain_tdvp)

println("Two-spin dissipative TDVP:")
println("  trace              = ", real(chain_tdvp_metrics.trace))
println("  hermiticity defect = ", chain_tdvp_metrics.hermiticity)
println("  min eigenvalue     = ", chain_tdvp_metrics.min_eig)

@assert ρL_chain_tdvp isa MPS{Liouville}
@assert abs(real(chain_tdvp_metrics.trace) - 1) < 1e-6
@assert chain_tdvp_metrics.hermiticity < 1e-6
@assert chain_tdvp_metrics.min_eig > -1e-6

# !!! note "TEBD and TDVP in this tutorial"
#     TEBD exposes the gate-based picture of `exp(𝓛Δt)`.
#     TDVP exposes the MPO-based picture of evolving inside an MPS manifold.
#     Both are useful once the density matrix has been written as
#     `MPS{Liouville}`.

# !!! tip "The important scaling pattern"
#     The single-spin and two-spin examples use the same command pattern:
#
#     ```julia
#     sites_L = liouv_sites(sites)
#     ρL0 = to_liouville(to_dm(ψ0); sites=sites_L)
#     ρL_t = tebd(ρL0, H, dt, T; jump_ops=jump_ops)
#     ```
#
#     The local Liouville dimension is larger, but the workflow remains
#     MPS/MPO-like.

# ### Summary
#
# In this tutorial, we learned that:
#
# - dissipative dynamics is naturally formulated for density matrices,
# - density matrices become `MPS{Liouville}` after vectorisation,
# - Liouvillian generators become `MPO{Liouville}`,
# - local jumps are passed as tuples such as `(γ, "S-", site)`,
# - `MPO_Liouville(H, sites_L; jump_ops=jump_ops)` builds the generator,
# - `tebd(ρL0, H, dt, T; jump_ops=jump_ops)` evolves the density matrix by
#   Liouville-space TEBD,
# - `tdvp(L_mpo, T, ρL0; time_step=dt)` evolves the same object using the
#   Liouville MPO,
# - trace, Hermiticity, and positivity are the basic sanity checks for
#   dissipative density-matrix dynamics.
#
# The next tutorial, [Single-Mode Process Tensor](@ref), moves beyond fixed
# Markovian Liouvillian generators. Process tensors describe reduced dynamics
# with memory, where the environment cannot be compressed into a time-local list
# of jump operators.
