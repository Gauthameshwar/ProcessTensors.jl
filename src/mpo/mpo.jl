# src/mpo/mpo.jl

# Import the functions from ITensorMPS that we want to extend
import ITensorMPS: MPO as CoreMPO

export MPO

struct MPO{S <: AbstractSpace, C} <: AbstractMPS{S}
    core::CoreMPO
    combiners::C
    
    function MPO{Hilbert}(core::CoreMPO)
        new{Hilbert, Nothing}(core, nothing)
    end
    
    function MPO{Liouville}(core::CoreMPO, combiners::Vector{ITensor})
        new{Liouville, Vector{ITensor}}(core, combiners)
    end
end

MPO(args...; kwargs...) = MPO{Hilbert}(CoreMPO(args...; kwargs...))
MPO(A::AbstractArray, args...; kwargs...) = MPO{Hilbert}(CoreMPO(A, args...; kwargs...))

MPO{Hilbert}(args...; kwargs...) = MPO{Hilbert}(CoreMPO(args...; kwargs...))
MPO{Hilbert}(A::AbstractArray, args...; kwargs...) = MPO{Hilbert}(CoreMPO(A, args...; kwargs...))

MPO{Liouville}(combiners::Vector{ITensor}, args...; kwargs...) = MPO{Liouville}(CoreMPO(args...; kwargs...), combiners)
MPO{Liouville}(combiners::Vector{ITensor}, A::AbstractArray, args...; kwargs...) = MPO{Liouville}(CoreMPO(A, args...; kwargs...), combiners)

copy(m::MPO{Hilbert}) = MPO{Hilbert}(copy(m.core))
copy(m::MPO{Liouville}) = MPO{Liouville}(copy(m.core), copy(m.combiners))
