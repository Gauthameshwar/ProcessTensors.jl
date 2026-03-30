# src/mps/constructors.jl

import ITensorMPS: outer, projector, state

# random_mps: wrap ITensorMPS.random_mps result into ProcessTensors.MPS
random_mps(sites::Vector{<:Index}; kwargs...) = MPS{Hilbert}(ITensorMPS.random_mps(sites; kwargs...))
random_mps(sites::Vector{<:Index}, state; kwargs...) = MPS{Hilbert}(ITensorMPS.random_mps(sites, state; kwargs...))
random_mps(eltype::Type{<:Number}, sites::Vector{<:Index}; kwargs...) = MPS{Hilbert}(ITensorMPS.random_mps(eltype, sites; kwargs...))
random_mps(eltype::Type{<:Number}, sites::Vector{<:Index}, state; kwargs...) = MPS{Hilbert}(ITensorMPS.random_mps(eltype, sites, state; kwargs...))

projector(m::AbstractMPS; kwargs...) = MPO{Hilbert}(ITensorMPS.projector(m.core; kwargs...))
outer(m1::AbstractMPS, m2::AbstractMPS; kwargs...) = MPO{Hilbert}(ITensorMPS.outer(m1.core, m2.core; kwargs...))