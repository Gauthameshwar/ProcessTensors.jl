# src/systems/systems.jl

import Base: show

"""
    AbstractSystem

Abstract interface for system models used in process-tensor construction.

Concrete systems store a Hamiltonian `H`, Lindblad jump operators `jump_ops`, and
canonical Liouville-space `sites` for the system degrees of freedom.
"""
abstract type AbstractSystem end

# Validate a system's site family and return canonical Liouville sites.
# `family_tag` is the SiteType substring required on every site (e.g. "S=" or "Boson");
# `family_desc` names the family in the error message (e.g. "spin indices").
# Hilbert sites are converted to Liouville; mixed Hilbert/Liouville inputs are rejected.
function _normalize_system_sites(
    sites::AbstractVector{<:Index},
    system_name::AbstractString,
    family_tag::AbstractString,
    family_desc::AbstractString,
)
    all(site -> any(t -> occursin(family_tag, t), tag_tokens(site)), sites) || throw(
        ArgumentError("$system_name: sites must be made of $family_desc. Got $(tag_tokens.(sites))."),
    )
    has_liouv = map(site -> has_tag_token(site, "Liouv"), sites)
    if any(has_liouv) && !all(has_liouv)
        throw(
            ArgumentError(
                "$system_name: all sites must be either all Hilbert (no 'Liouv' tag) or all Liouville (all have 'Liouv' tag). Got mixture: $(tag_tokens.(sites)).",
            ),
        )
    end
    any(has_liouv) || return liouv_sites(sites)
    return Index[sites...]
end

"""
    SpinSystem(sites, H, jump_ops)

Spin-system model for process-tensor construction.

`sites` may be either all Hilbert-space spin site indices or all Liouville-space
spin indices. Hilbert sites are converted with [`liouv_sites`](@ref); mixed
Hilbert/Liouville inputs are rejected. `H` is a physical Hamiltonian `OpSum`, and
`jump_ops` is a vector of Lindblad-channel `OpSum`s.
"""
struct SpinSystem <: AbstractSystem
    H::OpSum
    jump_ops::Vector{OpSum}
    sites::Vector{Index} # must be in the liouville space

    # Constructor to verify the indices input and H are all consistent
    function SpinSystem(sites::AbstractVector{<:Index}, H::OpSum, jump_ops::AbstractVector{<:OpSum})
        H == OpSum() && @warn "SpinSystem: H is empty. This is usually not what you want."
        liouv = _normalize_system_sites(sites, "SpinSystem", "S=", "spin indices")
        new(H, collect(jump_ops), liouv)
    end
end

"""
    BosonSystem(sites, H, jump_ops)

Bosonic-system model for process-tensor construction.

`sites` may be either all Hilbert-space boson site indices or all Liouville-space
boson indices. Hilbert sites are converted with [`liouv_sites`](@ref); mixed
Hilbert/Liouville inputs are rejected. `H` is a physical Hamiltonian `OpSum`, and
`jump_ops` is a vector of Lindblad-channel `OpSum`s.
"""
struct BosonSystem <: AbstractSystem
    H::OpSum
    jump_ops::Vector{OpSum}
    sites::Vector{Index}

    # Constructor to verify the indices input and H are all consistent
    function BosonSystem(sites::AbstractVector{<:Index}, H::OpSum, jump_ops::AbstractVector{<:OpSum})
        H == OpSum() && @warn "BosonSystem: H is empty. This is usually not what you want."
        liouv = _normalize_system_sites(sites, "BosonSystem", "Boson", "boson indices")
        new(H, collect(jump_ops), liouv)
    end
end

"""
    spin_system(sites, H; jump_ops=OpSum[])

Construct a [`SpinSystem`](@ref) using keyword-style Lindblad jump operators.
"""
spin_system(sites::AbstractVector{<:Index}, H::OpSum; jump_ops::AbstractVector{<:OpSum}=OpSum[]) =
    SpinSystem(sites, H, collect(jump_ops))

"""
    boson_system(sites, H; jump_ops=OpSum[])

Construct a [`BosonSystem`](@ref) using keyword-style Lindblad jump operators.
"""
boson_system(sites::AbstractVector{<:Index}, H::OpSum; jump_ops::AbstractVector{<:OpSum}=OpSum[]) =
    BosonSystem(sites, H, collect(jump_ops))

function Base.show(io::IO, sys::SpinSystem)
    ns = length(sys.sites)
    println(io, "ProcessTensors.SpinSystem")
    println(io, "  sites: ", ns)
    space = any(!has_tag_token(s, "Liouv") for s in sys.sites) ? "Hilbert" : "Liouville"
    println(io, "  space: ", space)
    site_dims = dim.(sys.sites)
    print(io, "  site dims: ")
    if length(site_dims) <= 10
        println(io, join(site_dims, ", "))
    else
        println(io, join(site_dims[1:5], ", "), ", ..., ", join(site_dims[(end - 4):end], ", "))
    end
    println(io, "  dissipative: ", !isempty(sys.jump_ops))
end

function Base.show(io::IO, sys::BosonSystem)
    ns = length(sys.sites)
    println(io, "ProcessTensors.BosonSystem")
    println(io, "  sites: ", ns)
    space = any(!has_tag_token(s, "Liouv") for s in sys.sites) ? "Hilbert" : "Liouville"
    println(io, "  space: ", space)
    site_dims = dim.(sys.sites)
    print(io, "  site dims: ")
    if length(site_dims) <= 10
        println(io, join(site_dims, ", "))
    else
        println(io, join(site_dims[1:5], ", "), ", ..., ", join(site_dims[(end - 4):end], ", "))
    end
    println(io, "  dissipative: ", !isempty(sys.jump_ops))
end

Base.show(io::IO, ::MIME"text/plain", sys::SpinSystem) = show(io, sys)
Base.show(io::IO, ::MIME"text/plain", sys::BosonSystem) = show(io, sys)
