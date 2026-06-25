# src/mpo/mpo.jl

import ITensorMPS: MPO as CoreMPO
import ITensorMPS: siteinds, linkdims, maxlinkdim
import Base: show, copy

"""
    AbstractMPO{S<:AbstractSpace}

Abstract interface for ProcessTensors matrix-product-operator wrappers.

`S` distinguishes ordinary Hilbert-space operators from Liouville-space
superoperators used in open-system and process-tensor contractions.
"""
abstract type AbstractMPO{S <: AbstractSpace} <: AbstractMPS{S} end

"""
    MPO{S<:AbstractSpace}

Matrix-product-operator wrapper around an `ITensorMPS.MPO` stored in `.core`.

`MPO{Hilbert}` represents an operator or density matrix on Hilbert-space site
indices and is the default result of `MPO(...)`. `MPO{Liouville}` represents a
Liouville-space superoperator acting on vectorized density matrices. Liouville
MPOs may carry `combiners` when constructed from fused Hilbert indices, matching
the convention used by [`to_liouville`](@ref) and [`MPO_Liouville`](@ref).

Most generic MPO operations are delegated to `.core` and rewrapped when they
return an MPS/MPO-like object.

# Examples
```julia
ρ = to_dm(random_mps(sites))
ρL = to_liouville(ρ; sites=liouv_sites(sites))  # MPS{Liouville} from density MPO
```
"""
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
MPO(A::ITensor, sites; kwargs...) = MPO{Hilbert}(CoreMPO(A, sites; kwargs...))

MPO{Hilbert}(args...; kwargs...) = MPO{Hilbert}(CoreMPO(args...; kwargs...))
MPO{Hilbert}(A::AbstractArray, args...; kwargs...) = MPO{Hilbert}(CoreMPO(A, args...; kwargs...))
MPO{Hilbert}(A::ITensor, sites; kwargs...) = MPO{Hilbert}(CoreMPO(A, sites; kwargs...))

MPO{Liouville}(combiners::Vector{ITensor}, args...; kwargs...) = MPO{Liouville}(CoreMPO(args...; kwargs...), combiners)
MPO{Liouville}(combiners::Vector{ITensor}, A::AbstractArray, args...; kwargs...) = MPO{Liouville}(CoreMPO(A, args...; kwargs...), combiners)

copy(m::MPO{Hilbert}) = MPO{Hilbert}(copy(m.core))
copy(m::MPO{Liouville}) = MPO{Liouville}(copy(m.core), copy(m.combiners))

function Base.show(io::IO, mpo::MPO{S, C}) where {S <: AbstractSpace, C}
    core = mpo.core
    N = length(core)
    println(io, "$N-element MPO{$S}")
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
    if mpo.combiners === nothing
        println(io, "none")
    else
        ncomb = length(mpo.combiners)
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
Base.show(io::IO, ::MIME"text/plain", mpo::MPO{S, C}) where {S <: AbstractSpace, C} =
    show(io, mpo)
