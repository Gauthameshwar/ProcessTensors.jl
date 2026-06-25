# src/mps/constructors.jl

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

# Tier C: `state`, `outer`, and `projector` follow ITensorMPS; wrappers return `MPS{Hilbert}` / `MPO{Hilbert}`.
outer(m1::AbstractMPS, m2::AbstractMPS; kwargs...) = MPO{Hilbert}(ITensorMPS.outer(m1.core, m2.core; kwargs...))
projector(m::AbstractMPS; kwargs...) = MPO{Hilbert}(ITensorMPS.projector(m.core; kwargs...))
