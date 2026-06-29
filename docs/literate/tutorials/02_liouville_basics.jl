# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors #src
# SPDX-License-Identifier: MIT #src
# #src
# File: docs/literate/tutorials/02_liouville_basics.jl #src
# Contributor: Gauthameshwar S. #src
# #src
# Literate tutorial source: density matrices, Liouville-space vectorisation, #src
# superoperators, and ProcessTensors.jl conventions. #src

# # Liouville-Space Basics
#
# This tutorial continues from [MPS and MPO Basics](@ref). We saw how a pure state
# $|\psi\rangle$ can be converted into a density matrix
# $\rho = |\psi\rangle\langle\psi|$ in Hilbert space.
# In this tutorial, we take the next step: we treat density matrices themselves
# as state-like vectors in a new space.
#
# Density matrices are more general than state vectors. They can describe
# coherent quantum superpositions, but also classical uncertainty:
#
# ```math
# \rho =
# \sum_i p_i |\psi_i\rangle\langle\psi_i|,
# \qquad
# p_i \geq 0,
# \qquad
# \sum_i p_i = 1.
# ```
#
# If a density matrix describes the state of a system, then a deterministic
# physical time evolution should map it to another valid density matrix:
#
# ```math
# \rho \longmapsto \Phi(\rho).
# ```
#
# Such maps are called _quantum channels_ when they are completely positive and
# trace-preserving (_CPTP_).
#
# In ordinary Hilbert-space notation, $\Phi$ looks like a function that takes a
# matrix as input. In Liouville space, the same map becomes an operator acting
# on a vectorized density matrix:
#
# ```math
# |\rho\rangle\rangle \longmapsto \Phi |\rho\rangle\rangle.
# ```
#
# This is why Liouville space is useful:
#
# > density matrices become state-like vectors, and quantum channels become
# > operator-like objects.

# ## Setup
#
# We load `ITensors`, `ITensorMPS`, and `ProcessTensors` for Hilbert/Liouville
# conversions and Liouvillian construction. The helper below contracts a
# one-site `MPO` into a dense matrix for cross-checks against package routines.

using ITensors
import LinearAlgebra
using ProcessTensors
import ITensorMPS

function dense_mpo_matrix(W, sites)
    T = ITensor(1.0)
    for n in eachindex(W)
        T *= W[n]
    end
    dims = dim.(sites)
    D = prod(dims)
    A = Array(T, prime.(sites)..., sites...)
    return reshape(ComplexF64.(A), D, D)
end

# ## Liouville states from vectorisation
#
# Let $\mathcal{H}$ be a finite-dimensional Hilbert space. The set of linear operators on
# $\mathcal{H}$ is written
#
# ```math
# \mathcal{B}(\mathcal{H}).
# ```
#
# A density matrix is an element of this operator space:
#
# ```math
# \rho \in \mathcal{B}(\mathcal{H}).
# ```
#
# The important observation is that this operator space is itself a Hilbert
# space when equipped with the Hilbert-Schmidt inner product:
#
# ```math
# \langle\langle A | B \rangle\rangle
# =
# \operatorname{Tr}(A^\dagger B).
# ```
#
# This operator Hilbert space is what we call **Liouville space**.
#
# In a chosen basis, we vectorize basis operators as
#
# ```math
# |j\rangle\langle k|
# \longmapsto
# |j\rangle \otimes |k\rangle.
# ```
#
# !!! note "Terminology"
#     In this package, a Hilbert-space density matrix is an `MPO{Hilbert}`.
#     After vectorization it becomes an `MPS{Liouville}`.

# ### A one-site density matrix
#
# We build a **complex** pure state so the resulting $\rho$ has distinct matrix
# elements. 
#
# ```math
# |\psi\rangle = 0.6\,|\uparrow\rangle + 0.8 i\,|\downarrow\rangle.
# ```

sites = siteinds("S=1/2", 1)
ψ = MPS(ComplexF64[0.6, 0.8im], sites)

ρ = to_dm(ψ)
ρ_dense = dense_mpo_matrix(ρ, sites)

println("Dense ρ from to_dm(ψ):")
println(ρ_dense)
println()
println("Tr(ρ)           = ", LinearAlgebra.tr(ρ_dense))
println("‖ρ - ρ†‖         = ", LinearAlgebra.norm(ρ_dense - ρ_dense'))
println("ρ₁₁, ρ₂₂ differ? = ", ρ_dense[1, 1] != ρ_dense[2, 2])
println("Off-diagonal ρ₁₂ = ", ρ_dense[1, 2])

@assert ρ isa MPO{Hilbert}
@assert isapprox(LinearAlgebra.tr(ρ_dense), 1.0 + 0im; atol=1e-12)
@assert isapprox(ρ_dense, ρ_dense'; atol=1e-12)
@assert !isapprox(ρ_dense[1, 1], ρ_dense[2, 2]; atol=1e-6)
@assert abs(imag(ρ_dense[1, 2])) > 1e-6

# ### Column-major vectorization
#
# Different communities use different vectorisation conventions. The convention
# matters because it determines the Kronecker-product formulas for left and right
# multiplication.
#
# `ProcessTensors.jl` uses **column-major vectorisation**, matching Julia's
# native array ordering. For a one-site density matrix
#
# ```math
# \rho =
# \begin{pmatrix}
# \rho_{11} & \rho_{12} \\
# \rho_{21} & \rho_{22}
# \end{pmatrix},
# ```
#
# the vectorised state is
#
# ```math
# |\rho\rangle\rangle =
# \operatorname{vec}(\rho) =
# \begin{pmatrix}
# \rho_{11} \\ \rho_{21} \\ \rho_{12} \\ \rho_{22}
# \end{pmatrix}.
# ```
#
# Equivalently, the **first matrix index changes fastest** when you read down
# the columns of $\rho$.

ρ_vec_dense = vec(ρ_dense)

println("Column-major vec(ρ):")
println(ρ_vec_dense)
println()
println("Compare with matrix entries:")
println("  vec[1] should equal ρ₁₁ = ", ρ_dense[1, 1])
println("  vec[2] should equal ρ₂₁ = ", ρ_dense[2, 1])
println("  vec[3] should equal ρ₁₂ = ", ρ_dense[1, 2])
println("  vec[4] should equal ρ₂₂ = ", ρ_dense[2, 2])

@assert isapprox(ρ_vec_dense[1], ρ_dense[1, 1]; atol=1e-12)
@assert isapprox(ρ_vec_dense[2], ρ_dense[2, 1]; atol=1e-12)
@assert isapprox(ρ_vec_dense[3], ρ_dense[1, 2]; atol=1e-12)
@assert isapprox(ρ_vec_dense[4], ρ_dense[2, 2]; atol=1e-12)

# !!! warning "Convention matters"
#     Row-major and column-major vectorization lead to different Kronecker
#     product formulas. This tutorial uses the column-major convention assumed
#     by `ProcessTensors.jl`.

# ### Liouville site indices and vectorization
#
# A spin-1/2 site has Hilbert dimension $d=2$ and Liouville dimension $d^2=4$.
# The function `liouv_sites(sites)` creates the enlarged local indices used by
# Liouville-space MPS and MPO objects.
#
# **Important:** create `sites_L = liouv_sites(sites)` once at the start of a
# workflow and reuse the same `Index` objects everywhere. Two separate calls to
# `liouv_sites` return different index objects even when their tags and
# dimensions match.

sites_L = liouv_sites(sites)

println("Hilbert site:   ", sites[1])
println("Liouville site: ", sites_L[1])
println("dim(Hilbert) = ", dim(sites[1]), ",  dim(Liouville) = ", dim(sites_L[1]))

@assert dim(sites_L[1]) == dim(sites[1])^2

ρL = to_liouville(ρ; sites=sites_L)
ρ_vec_pkg = ComplexF64.(Array(ρL[1], sites_L[1]))

println()
println("Vector from ProcessTensors.jl to_liouville:")
println(ρ_vec_pkg)
println()
println("Max |difference| with vec(ρ_dense) = ",
        maximum(abs.(ρ_vec_pkg - ρ_vec_dense)))

@assert ρL isa MPS{Liouville}
@assert isapprox(ρ_vec_pkg, ρ_vec_dense; atol=1e-12)

# The reverse operation is `to_hilbert`. It unzips the Liouville site back to the original Hilbert-space bra/ket MPO.

ρ_back = to_hilbert(ρL)
ρ_back_dense = dense_mpo_matrix(ρ_back, sites)

println()
println("Round-trip error ‖ρ - to_hilbert(to_liouville(ρ))‖ = ",
        LinearAlgebra.norm(ρ_back_dense - ρ_dense))

@assert isapprox(ρ_back_dense, ρ_dense; atol=1e-12)

# !!! note "What happened to the MPO?"
#     `to_liouville` fuses the ket and bra legs of an `MPO{Hilbert}` into one
#     enlarged Liouville leg. `to_hilbert` reverses that fusion. The round-trip
#     above confirms that no information is lost in this conversion on a
#     one-site system.

# ## Trace and expectation values
#
# Once operators and density matrices share the same Liouville index set, traces
# and expectations become Hilbert–Schmidt overlaps. In Liouville space,
# quantities such as $\operatorname{Tr}(\rho)$ and $\operatorname{Tr}(O\rho)$
# are computed as overlaps with vectorised operators. This is why identity
# operators and observable insertions appear naturally as contractions in
# tensor-network diagrams.

# ### Trace and local expectations
#
# The two central identities are
#
# ```math
# \operatorname{Tr}(\rho) = \langle\langle I | \rho \rangle\rangle,
# \qquad
# \langle O\rangle = \operatorname{Tr}(O\rho) = \langle\langle O | \rho \rangle\rangle.
# ```
#
# We build one-site Hilbert MPOs for the identity and $S_z$, vectorize them on
# the **same** `sites_L` used for $\rho$, and compare with dense-matrix formulas.

Id_os = OpSum()
Id_os += 1.0, "Id", 1
Id_mpo = MPO(Id_os, sites)

Sz_os = OpSum()
Sz_os += 1.0, "Sz", 1
Sz_mpo = MPO(Sz_os, sites)

Id_L = to_liouville(Id_mpo; sites=sites_L)
Sz_L = to_liouville(Sz_mpo; sites=sites_L)
Sz_dense = dense_mpo_matrix(Sz_mpo, sites)

trace_from_dense = LinearAlgebra.tr(ρ_dense)
trace_from_liouville = inner(Id_L, ρL)

expect_from_dense = LinearAlgebra.tr(Sz_dense * ρ_dense)
expect_from_liouville = inner(Sz_L, ρL)

println("Trace check:")
println("  Tr(ρ) from dense matrix:         ", trace_from_dense)
println("  ⟨⟨I|ρ⟩⟩ from Liouville overlap:  ", trace_from_liouville)
println()
println("Expectation value check:")
println("  Tr(Sz ρ) from dense matrices:       ", expect_from_dense)
println("  ⟨⟨Sz|ρ⟩⟩ from Liouville overlap:  ", expect_from_liouville)

@assert isapprox(trace_from_liouville, trace_from_dense; atol=1e-12)
@assert isapprox(expect_from_liouville, expect_from_dense; atol=1e-12)
@assert isapprox(expect_from_liouville, -0.14; atol=1e-12)

# ### Left and right multiplication
#
# A superoperator is a linear map acting on operators. After vectorisation, it
# becomes an ordinary operator acting on Liouville-space vectors. The central
# identity is
#
# ```math
# \operatorname{vec}(A\rho B) = (B^{\mathsf{T}}\otimes A)\operatorname{vec}(\rho).
# ```
#
# Two special cases are especially important:
#
# ```math
# \operatorname{vec}(A\rho) = (I\otimes A)|\rho\rangle\rangle,
# \qquad
# \operatorname{vec}(\rho B) = (B^{\mathsf{T}}\otimes I)|\rho\rangle\rangle.
# ```
#
# In package notation, `"Sx_L"` means left multiplication by $S_x$ and
# `"Sx_R"` means right multiplication by $S_x$ **before** vectorisation. The
# suffixes `_L` and `_R` do not mean “left tensor leg” and “right tensor leg”.

s = sites[1]
sL = sites_L[1]
A = Matrix(Array(op("Sx", s), prime(s), s))

A_left_dense = kron(Matrix{ComplexF64}(LinearAlgebra.I, 2, 2), A)
A_right_dense = kron(transpose(A), Matrix{ComplexF64}(LinearAlgebra.I, 2, 2))

left_expected = vec(A * ρ_dense)
right_expected = vec(ρ_dense * A)

left_from_kron = A_left_dense * ρ_vec_dense
right_from_kron = A_right_dense * ρ_vec_dense

println("Left multiplication Sx ρ:")
println("  ‖vec(Sx ρ) - (I⊗Sx) vec(ρ)‖ = ",
        LinearAlgebra.norm(left_expected - left_from_kron))
println()
println("Right multiplication ρ Sx:")
println("  ‖vec(ρ Sx) - (Sxᵀ⊗I) vec(ρ)‖ = ",
        LinearAlgebra.norm(right_expected - right_from_kron))

@assert LinearAlgebra.norm(left_expected - left_from_kron) < 1e-12
@assert LinearAlgebra.norm(right_expected - right_from_kron) < 1e-12

Sx_L_dense = Matrix(Array(op("Sx_L", sL), prime(sL), sL))
Sx_R_dense = Matrix(Array(op("Sx_R", sL), prime(sL), sL))

left_from_pkg = Sx_L_dense * ρ_vec_pkg
right_from_pkg = Sx_R_dense * ρ_vec_pkg

println()
println("Package Liouville operators:")
println("  ‖Sx_L |ρ⟩⟩ - vec(Sx ρ)‖ = ",
        LinearAlgebra.norm(left_from_pkg - left_expected))
println("  ‖Sx_R |ρ⟩⟩ - vec(ρ Sx)‖ = ",
        LinearAlgebra.norm(right_from_pkg - right_expected))

@assert LinearAlgebra.norm(left_from_pkg - left_expected) < 1e-12
@assert LinearAlgebra.norm(right_from_pkg - right_expected) < 1e-12

# !!! warning "Left and right are different"
#     `"Sx_L"` and `"Sx_R"` multiply the density matrix from different sides
#     before vectorization. They are not interchangeable.

# ## [Liouville superoperators and OpSums](@id liouville-superoperators-and-opsums)
#
# Closed-system evolution uses the commutator superoperator. Open-system
# generators add jump-operator terms in the same Liouville `OpSum` language.
# Once the density matrix is vectorised, Hamiltonian density-matrix evolution
# looks like ordinary linear evolution generated by a Liouville-space operator.

# ### Hamiltonian commutator generator
#
# Unitary Schrödinger evolution of a pure state becomes Liouville-space evolution
#
# ```math
# \frac{d\rho}{dt} = -i[H,\rho]
# \quad\Longleftrightarrow\quad
# \frac{d}{dt}|\rho\rangle\rangle = \mathcal{L}_H|\rho\rangle\rangle,
# ```
#
# where $\mathcal{L}_H = -iH_L + iH_R$ in the package suffix language. Here
# $H_L$ means “left multiplication by $H$” and $H_R$ means “right multiplication
# by $H$”.

H = OpSum()
H += 0.7, "Sx", 1
H += 0.2, "Sz", 1

L_os = OpSum_Liouville(H)

println()
println("Liouville-space OpSum for -i[H, .]:")
println(L_os)

L_mpo = MPO_Liouville(H, sites_L)

H_dense = dense_mpo_matrix(MPO(H, sites), sites)
dρ_vec_dense = vec(-1im * (H_dense * ρ_dense - ρ_dense * H_dense))
dρ_vec_pkg = ComplexF64.(Array(apply(L_mpo, ρL; cutoff=0.0)[1], sites_L[1]))

commutator_error = LinearAlgebra.norm(dρ_vec_pkg - dρ_vec_dense)

println("Hamiltonian commutator generator:")
println("  ‖package Liouvillian action - dense commutator‖ = ", commutator_error)

@assert commutator_error < 1e-10

# ### Open-system generator with a jump term
#
# Markovian open-system master equations add terms such as
# $L\rho L^\dagger - \tfrac{1}{2}\{L^\dagger L,\rho\}$.
# In `ProcessTensors.jl` a local jump is supplied as a tuple, e.g.
# `(γ, "S-", 1)`.

γ = 0.1
L_os_open = OpSum_Liouville(H; jump_ops=[(γ, "S-", 1)])

println("OpSum entries for the open-system Liouville generator:")
println(L_os_open)

L_mpo_open = MPO_Liouville(H, sites_L; jump_ops=[(γ, "S-", 1)])

println()
println("Open-system Liouville MPO built on the shared `sites_L` indices.")
println("MPO type: ", typeof(L_mpo_open))

# ### Common mistakes
#
# !!! warning "Do not recreate Liouville sites casually"
#     Two calls to `liouv_sites(sites)` return different `Index` objects even
#     when their tags and dimensions match.
#
# A frequent mistake is to vectorize a density matrix with one Liouville site
# set, then build `MPO_Liouville(H, sites)` from **Hilbert** sites. That
# constructor creates a fresh internal Liouville index that does not match the
# one already stored in `ρL`:

sites_L_for_state = liouv_sites(sites)
ρL_demo = to_liouville(ρ; sites=sites_L_for_state)
L_from_hilbert_sites = MPO_Liouville(H, sites)
sL_on_generator = only(siteinds(L_from_hilbert_sites))

println("Index on ρL:              ", sites_L_for_state[1])
println("Index on MPO_Liouville:   ", sL_on_generator)
println("Same index object?       ", sites_L_for_state[1] == sL_on_generator)

@assert sites_L_for_state[1] != sL_on_generator

try
    apply(L_from_hilbert_sites, ρL_demo)
    error("expected apply to fail on mismatched Liouville indices")
catch err
    println("apply fails as expected: ", typeof(err).name.name)
end

# Contractions such as `apply(L_from_hilbert_sites, ρL_demo)` then
# fail because ITensors matches legs by index identity, not by appearance.
# The fix is to pass the **same** Liouville sites to every constructor:
#
# ```julia
# sites_L = liouv_sites(sites)
# ρL = to_liouville(ρ; sites=sites_L)
# L_mpo = MPO_Liouville(H, sites_L)
# ```
#
# ### Summary
#
# - Liouville space vectorizes density matrices as `MPS{Liouville}`.
# - `ProcessTensors.jl` uses column-major vectorization and `liouv_sites`.
# - Traces and expectations are Liouville overlaps: `inner(O_L, ρL)`.
# - Left/right actions use `"Op_L"` / `"Op_R"` suffixes on Liouville sites.
# - `OpSum_Liouville` and `MPO_Liouville` build superoperators on **shared**
#   Liouville indices.
#
# In the next tutorial, [Unitary Dynamics](@ref), we use this machinery for
# unitary time evolution and compare Hilbert-space and Liouville-space dynamics.
