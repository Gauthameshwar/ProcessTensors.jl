# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: src/builders/abstract_builders.jl
# Contributor: Gauthameshwar S.
#
# Defines process-tensor builder interfaces and the dense builder selector.

"""
    AbstractPTBuilder

Abstract selector for process-tensor construction backends used by
[`build_process_tensor`](@ref).
"""
abstract type AbstractPTBuilder end

"""
    Dense()

Dense joint-Liouville process-tensor builder for no-bath, single-mode, and
small multimode environments.
"""
struct Dense <: AbstractPTBuilder end

"""
    ACE(; cutoff=1e-10, maxdim=typemax(Int), compression=:zipup_cpp, svd_alg=nothing)

Sequential automated-compression-of-environments (ACE) process-tensor builder
for baths of independent modes.

Joins closed single-mode bath process tensors onto the accumulated process
tensor one mode at a time. At each bond, `cutoff` is the relative ACE threshold
``ε``: retain singular values satisfying ``σᵢ > ε σ₁``. `maxdim` is an
additional safety cap on the retained bond dimension.

`compression` selects the join/truncation schedule:

- `:zipup_cpp` (default) follows the C++ ACE forward pass, truncating the
  current core before the next timestep of the incoming mode is joined, then
  sweeps backward.
- `:canonzip` joins one complete mode without truncation, moves the
  orthogonality center to the final time, then applies a right-to-left ACE SVD
  sweep.

`svd_alg` selects the LAPACK SVD on those memory bonds. `nothing` and
`"divide_and_conquer"` use LAPACK `gesdd` (the ITensors default).
`"qr_iteration"` uses LAPACK `gesvd`. `"recursive"` is the NDTensors
recursive fallback.

Requires `environment.coupling` to be empty; put every system-mode coupling on
the corresponding mode's `coupling` field.
"""
struct ACE <: AbstractPTBuilder
    cutoff::Float64
    maxdim::Int
    compression::Symbol
    svd_alg::Union{Nothing,String}

    function ACE(
        cutoff::Real,
        maxdim::Integer,
        compression::Symbol,
        svd_alg::Union{Nothing,AbstractString}=nothing,
    )
        cutoff >= 0 || throw(ArgumentError("ACE: cutoff must be non-negative; got $cutoff."))
        maxdim >= 1 || throw(ArgumentError("ACE: maxdim must be at least 1; got $maxdim."))
        compression in (:zipup_cpp, :canonzip) || throw(
            ArgumentError(
                "ACE: compression must be :zipup_cpp or :canonzip; got $compression." *
                (compression === :zipup ? " Join-ahead :zipup is removed; use :zipup_cpp." : ""),
            ),
        )
        alg = svd_alg === nothing ? nothing : String(svd_alg)
        alg in (nothing, "divide_and_conquer", "qr_iteration", "recursive") || throw(
            ArgumentError(
                "ACE: svd_alg must be nothing or \"divide_and_conquer\" (LAPACK gesdd), " *
                "\"qr_iteration\" (LAPACK gesvd), or \"recursive\"; got $(repr(svd_alg)).",
            ),
        )
        return new(float(cutoff), Int(maxdim), compression, alg)
    end
end

ACE(cutoff::Real, maxdim::Integer) = ACE(cutoff, maxdim, :zipup_cpp)

function ACE(;
    cutoff::Real=1e-10,
    maxdim::Integer=typemax(Int),
    compression::Symbol=:zipup_cpp,
    svd_alg::Union{Nothing,AbstractString}=nothing,
)
    return ACE(cutoff, maxdim, compression, svd_alg)
end
