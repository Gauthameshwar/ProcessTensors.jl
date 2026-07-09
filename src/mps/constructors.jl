# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: src/mps/constructors.jl
# Contributor: Gauthameshwar S.
#
# Provides MPS constructor helpers that forward to ITensorMPS and wrap results.

import ITensorMPS: outer, projector, state

"""
    random_mps(args...; kwargs...) -> MPS{Hilbert}

Construct a random Hilbert-space `MPS` by forwarding to `ITensorMPS.random_mps`
and wrapping the returned core.

# Examples
```julia
s = siteinds("S=1/2", 6)
ψ = random_mps(s; linkdims=4)
```
"""
random_mps(sites::Vector{<:Index}; kwargs...) = MPS{Hilbert}(ITensorMPS.random_mps(sites; kwargs...))
random_mps(sites::Vector{<:Index}, state; kwargs...) = MPS{Hilbert}(ITensorMPS.random_mps(sites, state; kwargs...))
random_mps(eltype::Type{<:Number}, sites::Vector{<:Index}; kwargs...) = MPS{Hilbert}(ITensorMPS.random_mps(eltype, sites; kwargs...))
random_mps(eltype::Type{<:Number}, sites::Vector{<:Index}, state; kwargs...) = MPS{Hilbert}(ITensorMPS.random_mps(eltype, sites, state; kwargs...))

outer(m1::AbstractMPS, m2::AbstractMPS; kwargs...) = MPO{Hilbert}(ITensorMPS.outer(m1.core, m2.core; kwargs...))
projector(m::AbstractMPS; kwargs...) = MPO{Hilbert}(ITensorMPS.projector(m.core; kwargs...))
