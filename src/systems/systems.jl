# src/systems/systems.jl

import Base: show

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

spin_system(sites::AbstractVector{<:Index}, H::OpSum; jump_ops::AbstractVector{<:OpSum}=OpSum[]) =
    SpinSystem(sites, H, collect(jump_ops))

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
