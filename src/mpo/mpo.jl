# src/mpo/mpo.jl

import ITensorMPS: MPO as CoreMPO

# The overarching abstract type for all ProcessTensors operator (MPO) objects
abstract type AbstractMPO{S <: AbstractSpace} <: AbstractMPS{S} end

struct MPO{S <: AbstractSpace, C} <: AbstractMPO{S}
    core::CoreMPO
    combiners::C
    
    function MPO{Hilbert}(core::CoreMPO)
        new{Hilbert, Nothing}(core, nothing)
    end
    
    function MPO{Liouville}(core::CoreMPO, combiners::Vector{ITensor})
        new{Liouville, Vector{ITensor}}(core, combiners)
    end
end
 
# Outer constructors
MPO(args...; kwargs...) = MPO{Hilbert}(CoreMPO(args...; kwargs...))
MPO(A::AbstractArray, args...; kwargs...) = MPO{Hilbert}(CoreMPO(A, args...; kwargs...))

MPO{Hilbert}(args...; kwargs...) = MPO{Hilbert}(CoreMPO(args...; kwargs...))
MPO{Hilbert}(A::AbstractArray, args...; kwargs...) = MPO{Hilbert}(CoreMPO(A, args...; kwargs...))

MPO{Liouville}(combiners::Vector{ITensor}, args...; kwargs...) = MPO{Liouville}(CoreMPO(args...; kwargs...), combiners)
MPO{Liouville}(combiners::Vector{ITensor}, A::AbstractArray, args...; kwargs...) = MPO{Liouville}(CoreMPO(A, args...; kwargs...), combiners)

copy(m::MPO{Hilbert}) = MPO{Hilbert}(copy(m.core))
copy(m::MPO{Liouville}) = MPO{Liouville}(copy(m.core), copy(m.combiners))
