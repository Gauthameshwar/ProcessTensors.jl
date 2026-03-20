# src/mps/mps.jl
using ..Basis: AbstractSpace, Hilbert, Liouville

import ITensors: ITensor
import ITensorMPS: AbstractMPS as CoreAbstractMPS
import ITensorMPS: MPS as CoreMPS
import Base: show, length, getindex, setindex!, copy

export AbstractMPS, MPS, AbstractSpace, Hilbert, Liouville, to_hilbert

# The overarching abstract type for all Process Tensor objects
abstract type AbstractMPS{S <: AbstractSpace} <: CoreAbstractMPS end

# Composite objects declared with `struct` are immutable
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

# Provide exact signature traps to bypass ITensorMPS Base.MPST injection ambiguity
MPS(args...; kwargs...) = MPS{Hilbert}(CoreMPS(args...; kwargs...))
MPS(A::AbstractArray, args...; kwargs...) = MPS{Hilbert}(CoreMPS(A, args...; kwargs...))

MPS{Hilbert}(args...; kwargs...) = MPS{Hilbert}(CoreMPS(args...; kwargs...))
MPS{Hilbert}(A::AbstractArray, args...; kwargs...) = MPS{Hilbert}(CoreMPS(A, args...; kwargs...))

MPS{Liouville}(combiners::Vector{ITensor}, args...; kwargs...) = MPS{Liouville}(CoreMPS(args...; kwargs...), combiners)
MPS{Liouville}(combiners::Vector{ITensor}, A::AbstractArray, args...; kwargs...) = MPS{Liouville}(CoreMPS(A, args...; kwargs...), combiners)

# Magic Property Delegation: Any missed fields (like `m.llim` or `m.rlim`) from ITensor algorithms drop directly into the core object!
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

# Often, one wants to customize how instances of a type are displayed, which is accomplished by overloading the show function
function show(io::IO, mps::AbstractMPS{S}) where {S <: AbstractSpace}
    print(io, "ProcessTensors.", typeof(mps).name.name, " { Space: ", S, " | Sites: ", length(mps.core), " }")
end

function to_hilbert(state::AbstractMPS{Liouville}) :: AbstractMPS{Hilbert}
    # 1. Grab the raw ITensor array from the core engine
    # (Assuming standard iteration over the core sites)
    raw_tensors = [state.core[i] for i in 1:length(state.core)]
    
    # 2. Use the attached combiners to "unzip" the indices back into 2D matrices
    unzipped_tensors = raw_tensors .* state.combiners 
    
    return MPS{Hilbert}(CoreMPS(unzipped_tensors))
end