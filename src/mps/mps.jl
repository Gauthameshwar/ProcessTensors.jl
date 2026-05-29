# src/mps/mps.jl

import ITensorMPS: AbstractMPS as CoreAbstractMPS
import ITensorMPS: MPS as CoreMPS
import ITensorMPS: siteinds, linkdims, maxlinkdim
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
MPS(A::ITensor, sites; kwargs...) = MPS{Hilbert}(CoreMPS(A, sites; kwargs...))

MPS{Hilbert}(args...; kwargs...) = MPS{Hilbert}(CoreMPS(args...; kwargs...))
MPS{Hilbert}(A::AbstractArray, args...; kwargs...) = MPS{Hilbert}(CoreMPS(A, args...; kwargs...))
MPS{Hilbert}(A::ITensor, sites; kwargs...) = MPS{Hilbert}(CoreMPS(A, sites; kwargs...))

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

function Base.show(io::IO, mps::MPS{S, C}) where {S <: AbstractSpace, C}
    core = mps.core
    N = length(core)
    println(io, "$N-element MPS{$S}")
    site_dims = Int[]
    for s in siteinds(core)
        push!(site_dims, dim(s isa Index ? s : first(s)))
    end
    ldims = collect(linkdims(core))
    print(io, "  site dims: ")
    if length(site_dims) <= 10
        print(io, join(site_dims, ", "))
    else
        print(io, join(site_dims[1:5], ", "), ", ..., ", join(site_dims[(end - 4):end], ", "))
    end
    println(io)
    print(io, "  link dims: ")
    if isempty(ldims)
        println(io, "none")
    elseif length(ldims) <= 10
        println(io, join(ldims, ", "))
    else
        println(io, join(ldims[1:5], ", "), ", ..., ", join(ldims[(end - 4):end], ", "))
    end
    println(io, "  maxlinkdim: ", maxlinkdim(core))
    print(io, "  combiners: ")
    if mps.combiners === nothing
        println(io, "none")
    else
        ncomb = length(mps.combiners)
        println(io, ncomb, " ITensor", ncomb == 1 ? "" : "s")
    end
    println(io)
    println(io, "  tensors:")
    for k in (N <= 6 ? (1:N) : (1:2))
        t = collect(inds(core[k]))
        print(io, "    [$k] ")
        if length(t) <= 10
            println(io, Tuple(t))
        else
            println(io, "(", join(string.(t[1:5]), ", "), ", ..., ", join(string.(t[(end - 4):end]), ", "), ")")
        end
    end
    if N > 6
        println(io, "    ⋮")
        for k in (N - 1):N
            t = collect(inds(core[k]))
            print(io, "    [$k] ")
            if length(t) <= 10
                println(io, Tuple(t))
            else
                println(io, "(", join(string.(t[1:5]), ", "), ", ..., ", join(string.(t[(end - 4):end]), ", "), ")")
            end
        end
    end
end

# REPL return-value display
Base.show(io::IO, ::MIME"text/plain", mps::MPS{S, C}) where {S <: AbstractSpace, C} =
    show(io, mps)
