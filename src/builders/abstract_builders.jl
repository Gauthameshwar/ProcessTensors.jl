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
    ACE(; cutoff=1e-10, maxdim=typemax(Int))

Sequential automated-compression-of-environments (ACE) process-tensor builder
for baths of independent modes.

Joins closed single-mode bath process tensors onto the accumulated process
tensor one mode at a time and compresses the memory bonds by SVD truncation
after each join (`cutoff` is the squared-singular-value truncation threshold,
`maxdim` caps the bond dimension). Requires `environment.coupling` to be empty;
put every system-mode coupling on the corresponding mode's `coupling` field.
"""
struct ACE <: AbstractPTBuilder
    cutoff::Float64
    maxdim::Int

    function ACE(cutoff::Real, maxdim::Integer)
        cutoff >= 0 || throw(ArgumentError("ACE: cutoff must be non-negative; got $cutoff."))
        maxdim >= 1 || throw(ArgumentError("ACE: maxdim must be at least 1; got $maxdim."))
        return new(float(cutoff), Int(maxdim))
    end
end

ACE(; cutoff::Real=1e-10, maxdim::Integer=typemax(Int)) = ACE(cutoff, maxdim)
