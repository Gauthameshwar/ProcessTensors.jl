# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors #src
# SPDX-License-Identifier: MIT #src
# #src
# File: docs/literate/tutorials/01_mps_mpo_basics.jl #src
# Contributor: Gauthameshwar S. #src
# #src
# Literate tutorial source: Hilbert-space MPS/MPO and density-matrix basics. #src

# # MPS and MPO Basics
#
# This tutorial continues from [ITensor Basics](@ref). This tutorial will walk you through
# how to define MPS, MPO, and density matrices in the Hilbert space, and how to compute
# expectation values and reduced density matrices.
# 
# For the original ITensorMPS tutorial on MPS and MPO objects, see the
# [MPS and MPO examples page](https://docs.itensor.org/ITensorMPS/stable/examples/MPSandMPO.html).
#
# A many-body wavefunction on `N` sites can be written as
#
# ```math
# |\psi\rangle =
# \sum_{\sigma_1,\ldots,\sigma_N}
# c_{\sigma_1\cdots\sigma_N}
# |\sigma_1,\ldots,\sigma_N\rangle.
# ```
#
# A matrix product state represents the coefficient tensor
#
# ```math
# c_{\sigma_1\cdots\sigma_N}
# ```
#
# as a product of smaller tensors:
#
# ```math
# c_{\sigma_1\cdots\sigma_N}
# =
# A^{\sigma_1}_1
# A^{\sigma_2}_2
# \cdots
# A^{\sigma_N}_N.
# ```
#
# This is useful because many physical states can be represented accurately
# using moderate bond dimensions.

# ## MPS from Tensors

# ### Setup
#
# We load `ITensors` for basic tensor operations and 
# import `ITensorMPS` only to inspect the object stored inside the
# `ProcessTensors.jl` wrappers.
#
# We use `ProcessTensors.jl` for the public MPS/MPO interface used throughout
# the rest of these tutorials.

using ITensors
import ITensorMPS
import LinearAlgebra
using ProcessTensors

roundreal(x; digits=6) = round(real(x); digits=digits)

# ### Decomposing a dense tensor into an MPS
#
# Let us first build a dense four-site wavefunction.
# Each site has dimension 2, so the full coefficient tensor has size
# `2 × 2 × 2 × 2`.
# We fill it with the numbers `1, 2, ..., 16`, then convert it into an MPS.

mps_sites = [Index(2, "Qubit,Site,n=$n") for n in 1:4]

coeffs = reshape(ComplexF64.(1:16), 2, 2, 2, 2)

ψ_dense = MPS(coeffs, mps_sites)

println(ψ_dense)

# Each ITensor core in this MPS contains physical indices (the `mps_sites`), and link indices connecting the cores. 
# One can access these indices from the MPS object using the `siteinds` and `linkinds` functions.
# 
# The MPS stores the same state as the dense tensor, but in factorized form. 
# To check this, we contract all MPS tensors back into one full tensor and compare the coefficients.

function dense_array_from_mps(ψ)
    full_tensor = ITensor(1.0)
    for n in eachindex(ψ)
        full_tensor *= ψ[n]
    end
    return Array(full_tensor, siteinds(ψ)...)
end

coeffs_from_mps = dense_array_from_mps(ψ_dense)

basis_index = (1, 2, 2, 1)

println("Dense coefficient at $basis_index: ", coeffs[basis_index...])
println("MPS coefficient at $basis_index:   ", coeffs_from_mps[basis_index...])

@assert isapprox(coeffs[basis_index...], coeffs_from_mps[basis_index...]; atol=1e-10)

# ### Physical site indices with `siteinds`
#
# For the sake of illustration, from now on, let us focus on a physical system and define everything in the context of spin-1/2 chains.
# The first ingredient we need to define a spin system wavefunction are the physical spin indices. 
# `siteinds("S=1/2", N)` creates the physical Hilbert-space indices for a spin chain of length `N`.

N = 6
sites = siteinds("S=1/2", N)

println("Spin-chain site indices:")
foreach(println, sites)

println("First spin site:")
println("  index = ", sites[1])
println("  dim   = ", dim(sites[1]))
println("  tags  = ", tags(sites[1]))

@assert length(sites) == N
@assert all(s -> dim(s) == 2, sites)

# !!! tip "Pro tip"
#     Create `sites` once and reuse them. Your states, operators, density
#     matrices, and later Liouville-space objects should all be built from
#     compatible site indices.

# !!! info "Beyond spin systems"
#     ITensorMPS has support to define spin, boson, fermion, electron, and other systems. 
#     See the [ITensorMPS SiteTypes page](https://docs.itensor.org/ITensorMPS/stable/IncludedSiteTypes.html) 
#     for the list of supported site types and how to create them.

# ### A product-state MPS
#
# Consider the Néel state
#
# ```math
# |\psi_{\mathrm{N\acute eel}}\rangle
# =
# |\uparrow\downarrow\uparrow\downarrow\uparrow\downarrow\rangle.
# ```
#
# In ITensorMPS syntax, this is very direct.

neel_config = ["Up", "Dn", "Up", "Dn", "Up", "Dn"]

ψ_neel = MPS(sites, neel_config)

println(ψ_neel)

# `ProcessTensors.jl` prints this as an `MPS{Hilbert}`.
#
# This means:
#
# - the object is an ordinary Hilbert-space MPS,
# - the underlying ITensorMPS object is stored in `.core`,
# - common MPS functions are forwarded to `.core`.

println("Wrapper type: ", typeof(ψ_neel))
println("Core type:    ", typeof(ψ_neel.core))

@assert ψ_neel isa MPS{Hilbert}
@assert ψ_neel.core isa ITensorMPS.MPS
@assert maxlinkdim(ψ_neel) == 1

# !!! info "Why does ProcessTensors.jl wrap MPS objects?"
#     The wrapper lets the package distinguish ordinary Hilbert-space states
#     from Liouville-space states later. For this tutorial, you can use the
#     wrapped object almost exactly like an ordinary `ITensorMPS.MPS`.

# Let us inspect one local MPS tensor and see if it contains the expected site and link indices.

middle = 3
println("Indices of ψ_neel[$middle]:")
println(inds(ψ_neel[middle]))

@assert siteind(ψ_neel, middle) in collect(inds(ψ_neel[middle]))

# ### Entanglement entropy of a pure MPS
#
# For a pure state
# $|\psi\rangle,$
# the entanglement entropy across a bipartition `A|B` is
#
# ```math
# S_A =
# -\operatorname{Tr}(\rho_A \log \rho_A),
# \qquad
# \rho_A =
# \operatorname{Tr}_B |\psi\rangle\langle\psi|.
# ```
#
# In MPS language, this entropy is computed from the singular values across a
# bond.
#
# For the product Néel state, the entanglement entropy across every cut should
# be zero.

S_neel = [entropy(ψ_neel, b) for b in 1:(N - 1)]

println("Bond entropies of the Néel state:")
println(roundreal.(S_neel))

@assert maximum(abs.(S_neel)) < 1e-12

# Let us also build a small GHZ state,
#
# ```math
# |\mathrm{GHZ}\rangle =
# \frac{1}{\sqrt{2}}
# \left(
# |\uparrow\uparrow\cdots\uparrow\rangle
# +
# |\downarrow\downarrow\cdots\downarrow\rangle
# \right).
# ```
#
# Every bipartition of this state has entropy `log(2)`.

function ghz_state(sites)
    N = length(sites)
    all_ups = fill("Up", N)
    all_dns = fill("Dn", N)
    ghz = (MPS(sites, all_ups) + MPS(sites, all_dns)) / sqrt(2)
    return ghz
end

ψ_ghz = ghz_state(sites)

S_ghz = [entropy(ψ_ghz, b) for b in 1:(N - 1)]

println("Bond entropies of the GHZ state:")
println(roundreal.(S_ghz))

@assert all(S -> isapprox(S, log(2); atol=1e-10), S_ghz)

# ## MPO and expectation values

# ### Local observables
#
# The local magnetization is
#
# ```math
# \langle S^z_j \rangle
# =
# \langle \psi | S^z_j | \psi \rangle.
# ```
#
# For the Néel state, the answer should alternate between `+1/2` and `-1/2`.
# The function `expect` performs the tensor-network contractions for all local
# one-site observables.

mz = expect(ψ_neel, "Sz")
mx = expect(ψ_neel, "Sx")

println("⟨Sz_j⟩:")
println(roundreal.(mz))

println("⟨Sx_j⟩:")
println(roundreal.(mx))

@assert isapprox.(real.(mz), [0.5, -0.5, 0.5, -0.5, 0.5, -0.5]; atol=1e-12) |> all
@assert all(abs.(mx) .< 1e-12)

# !!! note "Connection to the theory page"
#     This is the code version of the formula
#     `⟨O⟩ = ⟨ψ|O|ψ⟩`.
#
# The same expectation value can also be obtained by building the local operator
# explicitly as an MPO. For example, the one-site operator
#
# ```math
# O = S^z_3
# ```
#
# can be written as an `OpSum`, converted to an MPO, and contracted with the
# state.

function local_observable_mpo(opname::String, n::Integer, sites)
    os = OpSum()
    os += 1.0, opname, n
    return MPO(os, sites)
end

Sz3_mpo = local_observable_mpo("Sz", 3, sites)

sz3_from_mpo = inner(ψ_neel', Sz3_mpo, ψ_neel)

println("⟨Sz₃⟩ from local MPO: ", roundreal(sz3_from_mpo))
println("⟨Sz₃⟩ from expect:    ", roundreal(mz[3]))

@assert isapprox(sz3_from_mpo, mz[3]; atol=1e-12)

# Equivalently, we can apply the MPO to the state first:
#
# ```math
# |\phi\rangle = S^z_3 |\psi\rangle,
# ```
#
# then compute
#
# ```math
# \langle \psi | \phi \rangle.
# ```

ϕ = apply(Sz3_mpo, ψ_neel; cutoff=0.0)

sz3_from_apply = inner(ψ_neel, ϕ)

println("⟨ψ|Sz₃|ψ⟩ using apply: ", roundreal(sz3_from_apply))

@assert isapprox(sz3_from_apply, mz[3]; atol=1e-12)

# !!! note "What did `apply` return?"
#     Applying an MPO to an MPS returns another MPS. In `ProcessTensors.jl`, the
#     result is rewrapped as an `MPS{Hilbert}`.

@assert ϕ isa MPS{Hilbert}

# ### Building Hamiltonians with `OpSum`
#
# Now consider a simple spin-chain Hamiltonian:
#
# ```math
# H =
# h_x \sum_{j=1}^{N} S^x_j
# +
# J_z \sum_{j=1}^{N-1} S^z_j S^z_{j+1}.
# ```
#
# The `OpSum` object lets us write this in a physics-friendly way.

function spin_chain_opsum(N; hx=0.7, Jz=1.0)
    os = OpSum()
    for n in 1:N
        os += hx, "Sx", n
    end
    for n in 1:(N - 1)
        os += Jz, "Sz", n, "Sz", n + 1
    end
    return os
end

hx = 0.7
Jz = 1.0

H_os = spin_chain_opsum(N; hx, Jz)

println("Hamiltonian OpSum:")
println(H_os)

# !!! info "What is an OpSum?"
#     An `OpSum` is a symbolic/lazy sum of local operator terms. 
#     It contains only the metadata of what operator exists at which site, and with what coefficient. 
#     It is not yet an MPO. The MPO tensors are constructed when we call `MPO(H_os, sites)`.

# ### Matrix product operators
#
# A matrix product operator represents an operator as a tensor network:
#
# ```math
# \hat{O}
# =
# \sum_{\boldsymbol{\sigma},\boldsymbol{\sigma}'}
# W^{\sigma_1,\sigma'_1}_1
# W^{\sigma_2,\sigma'_2}_2
# \cdots
# W^{\sigma_N,\sigma'_N}_N
# |\boldsymbol{\sigma}\rangle
# \langle \boldsymbol{\sigma}'|.
# ```
#
# Constructing MPOs by hand can be difficult because the operator may need
# nontrivial internal bond dimensions.
#
# ITensorMPS has an efficient algorithm for constructing an MPO from an
# `OpSum`.

H_mpo = MPO(H_os, sites)

println(H_mpo)

println("MPO wrapper type: ", typeof(H_mpo))
println("MPO core type:    ", typeof(H_mpo.core))
println("MPO link dimensions: ", collect(linkdims(H_mpo)))

@assert H_mpo isa MPO{Hilbert}
@assert H_mpo.core isa ITensorMPS.MPO

# !!! tip "Pro tip"
#     Write operators as `OpSum`s whenever possible. Let ITensorMPS construct
#     the MPO cores for you. 
#     ITensorMPS has efficient algorithms to construct the MPO cores from the 
#     OpSum object with good accuracy and low memory overhead.

# ### Energy expectation value
#
# The energy of a state is
#
# ```math
# E =
# \langle \psi | H | \psi \rangle.
# ```
#
# In code:

E_neel = inner(ψ_neel', H_mpo, ψ_neel)

println("Energy of the Néel state: ", roundreal(E_neel))

# We can predict this value analytically.
#
# In the Néel state,
#
# ```math
# \langle S^x_j \rangle = 0
# ```
#
# and each nearest-neighbor pair is anti-aligned:
#
# ```math
# \langle S^z_j S^z_{j+1} \rangle = -\frac{1}{4}.
# ```
#
# Therefore
#
# ```math
# E =
# J_z (N-1)\left(-\frac{1}{4}\right).
# ```

E_expected = -Jz * (N - 1) / 4

println("Expected energy: ", E_expected)

@assert isapprox(real(E_neel), E_expected; atol=1e-12)

# ## Density matrices, reduced density matrices and entropy
#
# We now move from pure states to density matrices.
# A pure state has density matrix
#
# ```math
# \rho = |\psi\rangle\langle\psi|.
# ```
#
# A classical mixture of two pure states is
#
# ```math
# \rho =
# p |\psi_1\rangle\langle\psi_1|
# +
# (1-p) |\psi_2\rangle\langle\psi_2|.
# ```
#
# `ProcessTensors.jl` provides `to_dm` for exactly this purpose.

# ### A pure density MPO

ρ_neel = to_dm(ψ_neel)

println(ρ_neel)
println("Tr(ρ_neel) = ", roundreal(tr(ρ_neel)))

@assert ρ_neel isa MPO{Hilbert}
@assert isapprox(real(tr(ρ_neel)), 1.0; atol=1e-12)

# !!! info "Why is a density matrix an MPO?"
#     A density matrix has one ket index and one bra index per site. That makes
#     it naturally an operator-valued tensor network, i.e. an MPO.

# ### A mixed density MPO
#
# Let us mix two product states:
#
# ```math
# |\psi_1\rangle =
# |\uparrow\downarrow\uparrow\downarrow\cdots\rangle,
# ```
#
# and
#
# ```math
# |\psi_2\rangle =
# |\downarrow\uparrow\downarrow\uparrow\cdots\rangle.
# ```
#
# Define
#
# ```math
# \rho =
# p|\psi_1\rangle\langle\psi_1|
# +
# (1-p)|\psi_2\rangle\langle\psi_2|.
# ```

flipped_pattern = ["Dn", "Up", "Dn", "Up", "Dn", "Up"]
ψ_flipped = MPS(sites, flipped_pattern)

p = 0.3

ρ_mix = to_dm([ψ_neel, ψ_flipped]; coeffs=[p, 1 - p])

println("Tr(ρ_mix) = ", roundreal(tr(ρ_mix)))

@assert ρ_mix isa MPO{Hilbert}
@assert isapprox(real(tr(ρ_mix)), 1.0; atol=1e-12)

# ### One-site reduced density matrix
#
# Let us compute the reduced density matrix at one site.
#
# For a pure state,
#
# ```math
# \rho_j =
# \operatorname{Tr}_{\bar{j}}
# |\psi\rangle\langle\psi|.
# ```
#
# For an MPS, we can do this by constructing the pure density matrix and 
# contracting the bra/ket indices of the sites we wish to trace out.

function one_site_reduced_density_matrix(ρ, sites, site_number::Integer=1)
    ρ_tensor = ITensor(1.0)
    for n in eachindex(ρ)
        ρ_tensor *= ρ[n]
        if n != site_number
            ρ_tensor *= delta(sites[n], prime(sites[n]))
        end
    end
    return Array(ρ_tensor, prime(sites[site_number]), sites[site_number])
end

ρ1_red = one_site_reduced_density_matrix(ρ_mix, sites, 1)

@assert isapprox(LinearAlgebra.tr(ρ1_red), 1.0 + 0im; atol=1e-12)
display(roundreal.(ρ1_red))

# ### Physical entropy of the one-site mixed state
#
# The physical von Neumann entropy is
#
# ```math
# S(\rho) =
# -\operatorname{Tr}(\rho \log \rho).
# ```
#
# Since `ρ1_red` is a `2 × 2` dense matrix, we can compute its eigenvalues
# directly.

function von_neumann_entropy(ρ::AbstractMatrix)
    vals = real.(LinearAlgebra.eigvals(LinearAlgebra.Hermitian(ρ)))
    return -sum(λ -> λ > 1e-14 ? λ * log(λ) : 0.0, vals)
end

ρ1_mix = Matrix(ρ1_red)

S_ρ1_mix = von_neumann_entropy(ρ1_mix)

S_expected = -(p * log(p) + (1 - p) * log(1 - p))

println("S(ρ1_mix) = ", roundreal(S_ρ1_mix))
println("Expected binary entropy = ", roundreal(S_expected))

@assert isapprox(S_ρ1_mix, S_expected; atol=1e-12)

# !!! note "Mixed-state entropy is not pure-state entanglement"
#     The entropy above is physical von Neumann entropy of a one-site mixed
#     state. It contains classical uncertainty from the mixture. This is
#     generally different from the pure-state entanglement entropy computed 
#     from `entropy(ψ, b)`.

# ### MPO bond entropy as a compression diagnostic
#
# We can still call `entropy` on the density MPO.
#
# This computes an SVD-based bond entropy of the MPO tensor network. It tells us
# how much operator-space structure the density MPO carries across a cut.

middle_bond = div(N, 2)
S_operator_bond = entropy(ρ_mix, middle_bond)

println("MPO bond entropy of ρ_mix across bond $middle_bond:")
println(roundreal(S_operator_bond))

# ### Summary
#
# In this tutorial, we learned the basic ITensorMPS workflow used throughout
# `ProcessTensors.jl` for Hilbert-space states and operators.
#
# We saw that:
#
# - dense coefficient tensors can be decomposed into MPS form,
# - `siteinds` creates physical spin-chain site indices,
# - `MPS` creates Hilbert-space many-body states,
# - `OpSum` expresses many-body operators compactly,
# - `MPO` converts an `OpSum` into a matrix product operator,
# - `expect` computes local expectation values,
# - `correlation_matrix` computes two-point functions,
# - `entropy(ψ, b)` computes pure-state bond entanglement,
# - `to_dm` constructs Hilbert-space density MPOs.
#
# The main convention to remember is:
#
# > Reuse the correct indices. ITensors makes contractions safe because tensor
# > legs are labelled objects, not anonymous array axes.
#
# In the next tutorial, [Liouville-Space Basics](@ref), we take a density MPO and
# vectorize it into Liouville space.
