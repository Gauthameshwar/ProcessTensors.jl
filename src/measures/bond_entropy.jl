# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: src/measures/bond_entropy.jl
# Contributor: Gauthameshwar S.
#
# Defines the normalized bond entropy of an MPS, MPO, or process tensor, and
# the process-tensor wrapper temporal_entanglement.

import ITensorMPS: orthogonalize!, set_ortho_lims!
import ITensors: eigs, entropy, svd, uniqueinds

# `eigs(spec)` stores λ², so the von Neumann entropy uses those weights after dividing by their sum.
function _von_neumann_from_weights(w::AbstractVector{<:Real})
    z = sum(w)
    z > 0 || return 0.0
    S = 0.0
    for wi in w
        p = wi / z
        p > 0 && (S -= p * log(p))
    end
    return S
end

"""
    bond_entropy(W::AbstractMPS, b::Integer) -> Float64
    bond_entropy(W::AbstractMPS) -> Vector{Float64}

Von Neumann entropy, in nats, of the normalized squared Schmidt values across a virtual bond of `W`.

For a process tensor, this is the same as [`temporal_entanglement`](@ref). For spatial bonds it is the bipartite von Neumann entropy.

`bond_entropy(W, b)` orthogonalizes at core `b` and performs one SVD. `bond_entropy(W)` orthogonalizes once at core `1`, then takes one SVD per bond while moving the orthogonality center to the right. A tensor with fewer than two sites returns an empty vector.

# Arguments

- `W`: MPS, MPO, or process tensor.
- `b`: bond after site `b`, with `1 ≤ b ≤ length(W) - 1`.

# Examples

```julia
# `pt` is a ProcessTensor with `nsteps ≥ 2`.
S2 = bond_entropy(pt, 2)
profile = bond_entropy(pt)
```
"""
function bond_entropy end

function bond_entropy(W::AbstractMPS, b::Integer)
    N = length(W)
    1 <= b <= N - 1 || throw(
        ArgumentError("bond_entropy cut b must satisfy 1 ≤ b ≤ $(N - 1); got b=$b."),
    )
    core = W.core
    orthogonalize!(core, b)
    _, _, _, spec = svd(core[b], uniqueinds(core[b], core[b + 1]); cutoff=0.0)
    return _von_neumann_from_weights(eigs(spec))
end

function bond_entropy(W::AbstractMPS)
    core = W.core
    N = length(core)
    N >= 2 || return Float64[]
    orthogonalize!(core, 1)
    entropies = Vector{Float64}(undef, N - 1)
    for b in 1:(N - 1)
        U, S, V, spec = svd(core[b], uniqueinds(core[b], core[b + 1]); cutoff=0.0)
        entropies[b] = _von_neumann_from_weights(eigs(spec))
        core[b] = U
        core[b + 1] = (S * V) * core[b + 1]
        # Record the new orthogonality center so a later orthogonalize! does not skip QR.
        set_ortho_lims!(core, (b + 1):(b + 1))
    end
    return entropies
end

"""
    temporal_entanglement(pt::ProcessTensor, b::Integer) -> Float64
    temporal_entanglement(pt::ProcessTensor) -> Vector{Float64}

Temporal entanglement of a process tensor. The same value as [`bond_entropy`](@ref) on `pt`.
"""
function temporal_entanglement end

temporal_entanglement(pt::ProcessTensor, b::Integer) = bond_entropy(pt, b)
temporal_entanglement(pt::ProcessTensor) = bond_entropy(pt)

"""
    entropy(W::AbstractMPS, b::Integer) -> Float64

Bond entropy across one virtual bond. Equivalent to `bond_entropy(W, b)`.
"""
entropy(W::AbstractMPS, b::Integer) = bond_entropy(W, b)
