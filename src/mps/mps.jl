# src/mps/mps.jl

import ITensorMPS: AbstractMPS as CoreAbstractMPS
import ITensorMPS: MPS as CoreMPS
import Base: show, length, getindex, setindex!, copy

# The overarching abstract type for all ProcessTensors MPS/MPO objects
abstract type AbstractMPS{S <: AbstractSpace} <: CoreAbstractMPS end

struct MPS{S <: AbstractSpace, C} <: AbstractMPS{S}
    core::CoreMPS
    combiners::C
    
    function MPS{Hilbert}(core::CoreMPS)
        new{Hilbert, Nothing}(core, nothing)
    end
    
    function MPS{Liouville}(core::CoreMPS, combiners::Vector{ITensor})
        new{Liouville, Vector{ITensor}}(core, combiners)
    end
end

# Outer constructors
MPS(args...; kwargs...) = MPS{Hilbert}(CoreMPS(args...; kwargs...))
MPS(A::AbstractArray, args...; kwargs...) = MPS{Hilbert}(CoreMPS(A, args...; kwargs...))

MPS{Hilbert}(args...; kwargs...) = MPS{Hilbert}(CoreMPS(args...; kwargs...))
MPS{Hilbert}(A::AbstractArray, args...; kwargs...) = MPS{Hilbert}(CoreMPS(A, args...; kwargs...))

MPS{Liouville}(combiners::Vector{ITensor}, args...; kwargs...) = MPS{Liouville}(CoreMPS(args...; kwargs...), combiners)
MPS{Liouville}(combiners::Vector{ITensor}, A::AbstractArray, args...; kwargs...) = MPS{Liouville}(CoreMPS(A, args...; kwargs...), combiners)

# Property delegation: unknown fields drop through to the core object
function Base.getproperty(m::AbstractMPS, sym::Symbol)
    if sym === :core || sym === :combiners
        return getfield(m, sym)
    end
    return getproperty(getfield(m, :core), sym)
end

function Base.setproperty!(m::AbstractMPS, sym::Symbol, val)
    if sym === :core || sym === :combiners
        return setfield!(m, sym, val)
    end
    return setproperty!(getfield(m, :core), sym, val)
end

length(m::AbstractMPS) = length(m.core)
getindex(m::AbstractMPS, i::Integer) = m.core[i]
setindex!(m::AbstractMPS, v::ITensor, i::Integer) = (m.core[i] = v)

copy(m::MPS{Hilbert}) = MPS{Hilbert}(copy(m.core))
copy(m::MPS{Liouville}) = MPS{Liouville}(copy(m.core), copy(m.combiners))

function show(io::IO, mps::AbstractMPS{S}) where {S <: AbstractSpace}
    print(io, "ProcessTensors.", typeof(mps).name.name, " { Space: ", S, " | Sites: ", length(mps.core), " }")
end
