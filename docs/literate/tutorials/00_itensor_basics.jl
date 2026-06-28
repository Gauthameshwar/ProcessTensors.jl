# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors #src
# SPDX-License-Identifier: MIT #src
# #src
# File: docs/literate/tutorials/00_itensor_basics.jl #src
# Contributor: Gauthameshwar S. #src
# #src
# Literate tutorial source: ITensors indices, ITensors, contraction, and SVD. #src

# # ITensor Basics
#
# This tutorial introduces `Index` objects in ITensors, how they are used to 
# label tensor legs, and how ITensors contracts tensors by matching indices.
# 
# For the original ITensors tutorial on basic ITensor objects, see the
# [ITensor examples page](https://docs.itensor.org/ITensors/stable/examples/ITensor.html).

# ## Setup
#
# We load `ITensors` for basic tensor operations and `LinearAlgebra` for a
# dense singular-value check at the end.

using ITensors
import LinearAlgebra

roundreal(x; digits=6) = round(real(x); digits=digits)

# ## ITensor indices and objects
#
# A rank-3 tensor can be written as
#
# ```math
# T_{ijk}.
# ```
#
# The symbols `i`, `j`, and `k` label the three directions, or legs, of the
# tensor. In ordinary array language, we might say that `T` has three axes.
#
# For example, if
#
# ```math
# i = 1,2,\qquad
# j = 1,2,3,\qquad
# k = 1,2,
# ```
#
# then `T` has shape `2 × 3 × 2`.
#
# In ITensors, each of these labels becomes an `Index` object.

i = Index(2, "i")
j = Index(3, "j")
k = Index(2, "k")

println("Index i: ", i)
println("Index j: ", j)
println("Index k: ", k)

println("dim(i)  = ", dim(i))
println("tags(i) = ", tags(i))
println("plev(i) = ", plev(i))
println("id(i)   = ", id(i))

# !!! info "What is an ITensor Index?"
#     An `Index` is a labelled tensor leg. It has a dimension, tags, a prime
#     level, and an internal identity. Tensor contraction depends on index
#     identity, not just on dimension.

# ### Similar-looking indices are not necessarily the same
#
# Let us create another index with the same dimension and the same tags as `i`.

i_sim = sim(i)

println("Original index i: ", i)
println("Similar index:    ", i_sim)

println("dim(i) == dim(i_sim):   ", dim(i) == dim(i_sim))
println("tags(i) == tags(i_sim): ", tags(i) == tags(i_sim))
println("id(i) == id(i_sim):     ", id(i) == id(i_sim))
println("i == i_sim:             ", i == i_sim)

@assert dim(i) == dim(i_sim)
@assert tags(i) == tags(i_sim)
@assert i != i_sim

# !!! warning "Index identity matters"
#     Two indices with the same dimension and tags can still be different
#     indices. This is intentional. ITensors contracts tensors only when the
#     indices are the same objects, not merely when they look similar.
#
# This rule becomes extremely important later, especially in Liouville-space
# vectorization and process-tensor contractions.

# ### Creating an ITensor
#
# We now create a tensor
#
# ```math
# T_{ijk}
# ```
#
# with dimensions `2 × 3 × 2`.
#
# We fill it with the numbers `1, 2, ..., 12`.
A = reshape(1:12, 2, 3, 2)
T = ITensor(A, i, j, k)

println("T has indices:")
println(inds(T))

# Accessing a component looks like the mathematical notation:
#
# ```math
# T_{2,3,1}.
# ```

println("T[i=>2, j=>3, k=>1] = ", T[i => 2, j => 3, k => 1])

@assert T[i => 2, j => 3, k => 1] == 6

# !!! tip "Pro tip"
#     The order of the index-value pairs does not matter when accessing an
#     ITensor element. The labels identify the legs. This is in contrast with
#     the conventional methods where we need to worry about the order of the indices.

# You can also modify tensor entries.

println("Original T[1,1,1] = ", T[i => 1, j => 1, k => 1])

T[i => 1, j => 1, k => 1] = -100
println("Modified T[1,1,1] = ", T[i => 1, j => 1, k => 1])

@assert T[i => 1, j => 1, k => 1] == -100

# Convert the ITensor to an ordinary Julia array by specifying the index order.

T_array = Array(T, i, j, k)

println("Size of Array(T, i, j, k): ", size(T_array))
println("Array element [2,3,1]: ", T_array[2, 3, 1])
println("T[i => 2, j => 3, k => 1]: ", T[i => 2, j => 3, k => 1])

@assert T_array[2, 3, 1] == T[i => 2, j => 3, k => 1]

# !!! note "Why specify the index order?"
#     ITensors are not meant to be treated as arrays with a hidden axis order.
#     When you convert an ITensor into an array, you explicitly say which index
#     should become the first, second, third, ... array axis.

# ## Contractions and the SVD

# ### Contracting tensor indices
#
# Tensor contraction means summing over shared indices.
#
# For example, if
#
# ```math
# A_{ij}
# ```
#
# and
#
# ```math
# B_{jk}
# ```
#
# share the index `j`, then their contraction is
#
# ```math
# C_{ik}
# =
# \sum_j A_{ij} B_{jk}.
# ```
#
# In ITensors, this contraction is simply multiplication.

a = Index(2, "a")
b = Index(3, "b")
c = Index(4, "c")

A = random_itensor(a, b)
B = random_itensor(b, c)

C = A * B

println("inds(A) = ", inds(A))
println("inds(B) = ", inds(B))
println("inds(C) = ", inds(C))

@assert a in collect(inds(C))
@assert c in collect(inds(C))
@assert !(b in collect(inds(C)))

# The shared index `b` disappeared because it was summed over.

# ### Tracing two legs with a delta tensor
#
# We can also contract two legs of the same tensor using a Kronecker delta.
#
# Suppose
#
# ```math
# T_{ijk}
# ```
#
# has `dim(i) = dim(k)`. Then
#
# ```math
# v_j
# =
# \sum_{a=1}^{\dim(i)} T_{aja}
# ```
#
# is obtained by inserting a delta tensor
#
# ```math
# \delta_{ik}.
# ```
#
# In code:

δik = delta(i, k)
v = δik * T

println("inds(delta(i,k) * T) = ", inds(v))
println("v as an array over j:")
println(Array(v, j))

# Let us check the formula manually.

manual_v = zeros(Float64, dim(j))
for jj in 1:dim(j)
    manual_v[jj] = T[i => 1, j => jj, k => 1] + T[i => 2, j => jj, k => 2]
end

@assert Array(v, j) ≈ manual_v

# !!! tip "Pro tip"
#     A delta tensor is often the cleanest way to express traces, index
#     identifications, and partial traces in ITensor code.

# ### Tensor SVD
#
# Singular value decomposition is the basic operation behind tensor-network
# compression.
#
# For a matrix,
#
# ```math
# M = U S V^\dagger.
# ```
#
# For a tensor, we first decide which indices belong to the left group and which
# belong to the right group.
#
# Here we view
#
# ```math
# T_{ijk}
# ```
#
# as a matrix
#
# ```math
# T_{(ij),k}.
# ```
#
# Then we decompose it as
#
# ```math
# T_{ijk}
# =
# \sum_{\alpha}
# U_{ij,\alpha}
# S_{\alpha}
# V_{\alpha,k}.
# ```

U, S, V, spec = svd(T, (i, j))

T_reconstructed = U * S * V

println("Reconstruction error after SVD:")
println(norm(T - T_reconstructed))

@assert norm(T - T_reconstructed) < 1e-10

# The singular values are the singular values of the reshaped matrix
# `T[(i,j), k]`.

T_matrix = reshape(Array(T, i, j, k), dim(i) * dim(j), dim(k))
svals_dense = LinearAlgebra.svdvals(T_matrix)

println("Singular values of T[(i,j),k]:")
println(roundreal.(svals_dense))

# !!! info "Why SVD matters"
#     Tensor SVD is the local move behind MPS construction, MPS compression,
#     MPO compression, TEBD, TDVP, and the memory compression ideas that appear
#     later in process-tensor algorithms.
#
# **Next:** [MPS and MPO Basics](@ref) applies these ideas to many-body states
# and operators on spin chains.
