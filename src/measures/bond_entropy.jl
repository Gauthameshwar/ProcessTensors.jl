# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: src/measures/bond_entropy.jl
# Contributor: Gauthameshwar S.
#
# Defines the normalized Schmidt entropy across one virtual bond, or across
# every bond, of an MPS, MPO, or process tensor.

import ITensorMPS: orthogonalize!, set_ortho_lims!
import ITensors: eigs, svd, uniqueinds

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

Von Neumann entropy, in nats, of the Schmidt spectrum across a virtual bond of `W`.

For a [`ProcessTensor`](@ref), site `b` is time step `b`. The cut after core `b`
separates intervals `1:b` from `b+1:nsteps`. The result is the operator-space
entanglement of the stored MPO, not the von Neumann entropy of the reduced
system state. An [`ACE`](@ref) process tensor reports the spectrum that remains
after compression.

The weights are the squared Schmidt values returned by `eigs` of one SVD at the
cut, divided by their sum. Both methods bring `W.core` to a canonical gauge with
`orthogonalize!` and modify that gauge in place. The represented tensor is
unchanged. Site indices are preserved. Link indices may be replaced.

`bond_entropy(W, b)` orthogonalizes at core `b` and performs one SVD.
`bond_entropy(W)` orthogonalizes once at core `1`, then takes one SVD per bond
while moving the orthogonality center to the right. A tensor with fewer than
two sites returns an empty vector.

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
