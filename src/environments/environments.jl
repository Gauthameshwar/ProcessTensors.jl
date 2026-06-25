# src/environments/environments.jl

module Environments

using ..ProcessTensors: AbstractMPS, MPS, Hilbert, Liouville, OpSum, Index, siteinds, has_tag_token,
                       MAX_DENSE_LIOUVILLE_DIM
using ..Spectrals: AbstractSpectralDensity, ohmic_sd
using ITensors
using ITensors: dim, terms
import Base: show

export AbstractBathMode, AbstractBath, BosonicMode, SpinMode, BosonicBath, SpinBath,
       bosonic_mode, spin_mode, bosonic_bath, spin_bath,
       mode_initial_states

"""
    AbstractBathMode

Abstract interface for a single bath mode coupled to a system.

Concrete modes store an initial state `rho0`, a local Hamiltonian `H`, optional
mode-system `coupling`, and Liouville-space `sites`.
"""
abstract type AbstractBathMode end

"""
    AbstractBath

Abstract interface for bath containers used by [`build_process_tensor`](@ref ProcessTensors.build_process_tensor).

Concrete baths group compatible bath modes, an optional spectral-density model,
inter-mode coupling data, and the Liouville-space bath sites.
"""
abstract type AbstractBath end

"""
    BosonicMode(sites, H, n_max, rho0; coupling=OpSum())
    BosonicMode(sites, H, rho0; n_max=dim(only(sites))-1, coupling=OpSum())

Single bosonic bath mode in Liouville space.

`rho0` is the initial mode state and must have site indices exactly equal to
`sites`. `H` is the local mode Hamiltonian. `coupling` uses local site labels
with site `1` for the bath mode and site `2` for the coupled system site.
"""
struct BosonicMode{M<:AbstractMPS} <: AbstractBathMode
    rho0::M # must have liouville index space
    H::OpSum
    coupling::OpSum # local pair: site 1 = bath, site 2 = system
    n_max::Int
    sites::Vector{Index} # must be in liouville space

    # Constructor to verify the indices input, n_max, and H are all consistent
    function BosonicMode{M}(
        sites::AbstractVector{<:Index},
        H::OpSum,
        n_max::Int,
        rho0::M,
        coupling::OpSum=OpSum(),
    ) where {M<:AbstractMPS}
        length(sites) == 1 || throw(ArgumentError("BosonicMode: a single bosonic mode should have exactly one site index. Got $(length(sites))."))
        siteinds(rho0) == sites || throw(ArgumentError("BosonicMode:rho0 and sites must have the same indices. Got $(siteinds(rho0)) and $(sites)."))
        n_max == dim(only(sites)) - 1 || throw(ArgumentError("BosonicMode:n_max must be dim(sites) - 1. Got $n_max for sites with dim=$(dim(only(sites)))."))
        H == OpSum() && @warn "BosonicMode:H is empty. This is usually not what you want."
        new(rho0, H, coupling, n_max, Index[sites...])
    end
end

function BosonicMode(sites::AbstractVector{<:Index}, H::OpSum, n_max::Int, rho0::M; coupling::OpSum=OpSum()) where {M<:AbstractMPS}
    return BosonicMode{M}(sites, H, n_max, rho0, coupling)
end

BosonicMode(sites::AbstractVector{<:Index}, H::OpSum, rho0::AbstractMPS; n_max::Int=dim(only(sites)) - 1, coupling::OpSum=OpSum()) =
    BosonicMode(sites, H, n_max, rho0; coupling=coupling)

BosonicMode(; sites::AbstractVector{<:Index}, H::OpSum=OpSum(), rho0::AbstractMPS, n_max::Int=dim(only(sites)) - 1, coupling::OpSum=OpSum()) =
    BosonicMode(sites, H, n_max, rho0; coupling=coupling)

"""
    SpinMode(sites, H, rho0; coupling=OpSum())

Single spin bath mode in Liouville space.

`rho0` is the initial mode state and must have site indices exactly equal to
`sites`. `H` is the local mode Hamiltonian. `coupling` uses local site labels
with site `1` for the bath mode and site `2` for the coupled system site.
"""
struct SpinMode{M<:AbstractMPS} <: AbstractBathMode
    rho0::M # must have liouville index space
    H::OpSum
    coupling::OpSum # local pair: site 1 = bath, site 2 = system
    sites::Vector{Index} # must be in liouville space

    # Constructor to verify the indices input and H are all consistent
    function SpinMode{M}(sites::AbstractVector{<:Index}, H::OpSum, rho0::M; coupling::OpSum=OpSum()) where {M<:AbstractMPS}
        length(sites) == 1 || throw(ArgumentError("SpinMode: a single spin mode should have exactly one site index. Got $(length(sites))."))
        siteinds(rho0) == sites || throw(ArgumentError("SpinMode:rho0 and sites must have the same indices. Got $(siteinds(rho0)) and $(sites)."))
        H == OpSum() && @warn "SpinMode:H is empty. This is usually not what you want."
        new(rho0, H, coupling, Index[sites...])
    end
end

function SpinMode(sites::AbstractVector{<:Index}, H::OpSum, rho0::AbstractMPS; coupling::OpSum=OpSum())
    return SpinMode{typeof(rho0)}(sites, H, rho0; coupling=coupling)
end

SpinMode(; sites::AbstractVector{<:Index}, H::OpSum=OpSum(), rho0::AbstractMPS, coupling::OpSum=OpSum()) =
    SpinMode(sites, H, rho0; coupling=coupling)

"""
    BosonicBath(modes, spectral_density=ohmic_sd(), coupling=OpSum())
    BosonicBath(sites, modes, spectral_density, coupling)

Bath container for bosonic modes.

`modes` must contain [`BosonicMode`](@ref) values. `sites` are the concatenated
Liouville sites of the modes. `coupling` represents bath-only inter-mode terms;
mode-system couplings belong on each mode's `coupling` field.
"""
struct BosonicBath{M<:BosonicMode,S<:AbstractSpectralDensity,O<:OpSum} <: AbstractBath
    modes::Vector{M}
    spectral_density::S
    coupling::O
    sites::Vector{Index}

    # Constructor to verify the indices input, spectral_density, and coupling are all consistent
    function BosonicBath{M,S,O}(
        sites::AbstractVector{<:Index},
        modes::AbstractVector{M},
        spectral_density::S,
        coupling::O,
    ) where {M<:BosonicMode,S<:AbstractSpectralDensity,O<:OpSum}
        all(mode -> mode isa BosonicMode, modes) || throw(ArgumentError("BosonicBath:modes must contain only BosonicMode values."))
        length(modes) == length(sites) || throw(ArgumentError("BosonicBath:modes and sites must have the same length. Got $(length(modes)) and $(length(sites))."))
        all(mode -> mode.coupling == OpSum(), modes) &&
            coupling == OpSum() &&
            @warn "BosonicBath: no mode-system coupling on modes or inter-mode coupling on bath. This is usually not what you want."
        d_bath = isempty(sites) ? 1 : prod(dim.(collect(sites)))
        if d_bath > MAX_DENSE_LIOUVILLE_DIM
            @warn "BosonicBath has bath-only Liouville dimension D_bath=$d_bath (> $MAX_DENSE_LIOUVILLE_DIM). " *
                  "It will fail in dense build_process_tensor once the system coupling site is included."
        end
        new(collect(modes), spectral_density, coupling, Index[sites...])
    end
end

function BosonicBath(
    sites::AbstractVector{<:Index},
    modes::AbstractVector{M},
    spectral_density::S,
    coupling::O,
) where {M<:BosonicMode,S<:AbstractSpectralDensity,O<:OpSum}
    return BosonicBath{M,S,O}(sites, modes, spectral_density, coupling)
end
function BosonicBath(
    modes::AbstractVector{<:BosonicMode},
    spectral_density::AbstractSpectralDensity=ohmic_sd(),
    coupling::OpSum=OpSum(),
)
    sites = collect(Iterators.flatten(getfield.(modes, :sites)))
    return BosonicBath(sites, modes, spectral_density, coupling)
end
function BosonicBath(; modes::AbstractVector=BosonicMode[], spectral_density::AbstractSpectralDensity=ohmic_sd(), coupling::OpSum=OpSum())
    all(mode -> mode isa BosonicMode, modes) || throw(ArgumentError("BosonicBath:modes must contain only BosonicMode values."))
    return BosonicBath(collect(BosonicMode, modes), spectral_density, coupling)
end

"""
    SpinBath(modes, spectral_density=ohmic_sd(), coupling=OpSum())
    SpinBath(sites, modes, spectral_density, coupling)

Bath container for spin modes.

`modes` must contain [`SpinMode`](@ref) values. `sites` are the concatenated
Liouville sites of the modes. `coupling` represents bath-only inter-mode terms;
mode-system couplings belong on each mode's `coupling` field.
"""
struct SpinBath{M<:SpinMode,S<:AbstractSpectralDensity,O<:OpSum} <: AbstractBath
    modes::Vector{M}
    spectral_density::S
    coupling::O
    sites::Vector{Index}

    # Constructor to verify the indices input, spectral_density, and coupling are all consistent
    function SpinBath{M,S,O}(
        sites::AbstractVector{<:Index},
        modes::AbstractVector{M},
        spectral_density::S,
        coupling::O,
    ) where {M<:SpinMode,S<:AbstractSpectralDensity,O<:OpSum}
        all(mode -> mode isa SpinMode, modes) || throw(ArgumentError("SpinBath: modes must contain only SpinMode values."))
        length(modes) == length(sites) || throw(ArgumentError("SpinBath: modes and sites must have the same length. Got $(length(modes)) and $(length(sites))."))
        all(mode -> mode.coupling == OpSum(), modes) &&
            coupling == OpSum() &&
            @warn "SpinBath: no mode-system coupling on modes or inter-mode coupling on bath. This is usually not what you want."
        d_bath = isempty(sites) ? 1 : prod(dim.(collect(sites)))
        if d_bath > MAX_DENSE_LIOUVILLE_DIM
            @warn "SpinBath has bath-only Liouville dimension D_bath=$d_bath (> $MAX_DENSE_LIOUVILLE_DIM). " *
                  "It will fail in dense build_process_tensor once the system coupling site is included."
        end
        new(collect(modes), spectral_density, coupling, Index[sites...])
    end
end

function SpinBath(
    sites::AbstractVector{<:Index},
    modes::AbstractVector{M},
    spectral_density::S,
    coupling::O,
) where {M<:SpinMode,S<:AbstractSpectralDensity,O<:OpSum}
    return SpinBath{M,S,O}(sites, modes, spectral_density, coupling)
end
function SpinBath(
    modes::AbstractVector{<:SpinMode},
    spectral_density::AbstractSpectralDensity=ohmic_sd(),
    coupling::OpSum=OpSum(),
)
    sites = collect(Iterators.flatten(getfield.(modes, :sites)))
    return SpinBath(sites, modes, spectral_density, coupling)
end
function SpinBath(; modes::AbstractVector=SpinMode[], spectral_density::AbstractSpectralDensity=ohmic_sd(), coupling::OpSum=OpSum())
    all(mode -> mode isa SpinMode, modes) || throw(ArgumentError("SpinBath:modes must contain only SpinMode values."))
    return SpinBath(collect(SpinMode, modes), spectral_density, coupling)
end

"""
    bosonic_mode(args...; kwargs...)

Lowercase convenience constructors for [`BosonicMode`](@ref) and [`SpinMode`](@ref).
"""
bosonic_mode(args...; kwargs...) = BosonicMode(args...; kwargs...)

"""
    spin_mode(args...; kwargs...)

Lowercase convenience constructor for [`SpinMode`](@ref).
"""
spin_mode(args...; kwargs...) =
    SpinMode(args...; kwargs...)

"""
    bosonic_bath(modes; spectral_density=ohmic_sd(), coupling=OpSum())

Lowercase convenience constructors for [`BosonicBath`](@ref) and [`SpinBath`](@ref).
"""
bosonic_bath(modes::AbstractVector{<:BosonicMode}; spectral_density::AbstractSpectralDensity=ohmic_sd(), coupling::OpSum=OpSum()) =
    BosonicBath(modes, spectral_density, coupling)
bosonic_bath(; modes::AbstractVector=BosonicMode[], spectral_density::AbstractSpectralDensity=ohmic_sd(), coupling::OpSum=OpSum()) =
    BosonicBath(; modes=modes, spectral_density=spectral_density, coupling=coupling)

"""
    spin_bath(modes; spectral_density=ohmic_sd(), coupling=OpSum())

Lowercase convenience constructor for [`SpinBath`](@ref).
"""
spin_bath(modes::AbstractVector{<:SpinMode}; spectral_density::AbstractSpectralDensity=ohmic_sd(), coupling::OpSum=OpSum()) =
    SpinBath(modes, spectral_density, coupling)
spin_bath(; modes::AbstractVector=SpinMode[], spectral_density::AbstractSpectralDensity=ohmic_sd(), coupling::OpSum=OpSum()) =
    SpinBath(; modes=modes, spectral_density=spectral_density, coupling=coupling)

"""
    mode_initial_states(bath::AbstractBath)

Return the initial state of each bath mode in `bath`.
"""
mode_initial_states(bath::AbstractBath) = getfield.(bath.modes, :rho0)

function Base.show(io::IO, mode::BosonicMode)
    println(io, "ProcessTensors.BosonicMode")
    space = any(!has_tag_token(s, "Liouv") for s in mode.sites) ? "Hilbert" : "Liouville"
    println(io, "  space: ", space)
    site_dims = dim.(mode.sites)
    print(io, "  site dims: ")
    if length(site_dims) <= 10
        println(io, join(site_dims, ", "))
    else
        println(io, join(site_dims[1:5], ", "), ", ..., ", join(site_dims[(end - 4):end], ", "))
    end
    ρ = mode.rho0
    initial = ρ isa MPS{Liouville} ? "MPS{Liouville}" :
              ρ isa MPS{Hilbert} ? "MPS{Hilbert}" : string(typeof(ρ))
    println(io, "  initial state: ", initial)
    print(io, "  coupling: ")
    if isempty(terms(mode.coupling))
        println(io, "none")
    else
        labels = String[]
        for t in terms(mode.coupling)
            label = replace(sprint(show, t), r"\(\d+,?\)" => "")
            label = replace(replace(strip(label), " " => ""), r"^[-+]?[\d\.]+" => "")
            push!(labels, label)
        end
        println(io, '"', join(labels, "+"), '"')
    end
    println(io, "  n_max: ", mode.n_max)
end

function Base.show(io::IO, mode::SpinMode)
    println(io, "ProcessTensors.SpinMode")
    space = any(!has_tag_token(s, "Liouv") for s in mode.sites) ? "Hilbert" : "Liouville"
    println(io, "  space: ", space)
    site_dims = dim.(mode.sites)
    print(io, "  site dims: ")
    if length(site_dims) <= 10
        println(io, join(site_dims, ", "))
    else
        println(io, join(site_dims[1:5], ", "), ", ..., ", join(site_dims[(end - 4):end], ", "))
    end
    ρ = mode.rho0
    initial = ρ isa MPS{Liouville} ? "MPS{Liouville}" :
              ρ isa MPS{Hilbert} ? "MPS{Hilbert}" : string(typeof(ρ))
    println(io, "  initial state: ", initial)
    print(io, "  coupling: ")
    if isempty(terms(mode.coupling))
        println(io, "none")
    else
        labels = String[]
        for t in terms(mode.coupling)
            label = replace(sprint(show, t), r"\(\d+,?\)" => "")
            label = replace(replace(strip(label), " " => ""), r"^[-+]?[\d\.]+" => "")
            push!(labels, label)
        end
        println(io, '"', join(labels, "+"), '"')
    end
end

function Base.show(io::IO, bath::BosonicBath)
    nm = length(bath.modes)
    println(io, "ProcessTensors.BosonicBath")
    println(io, "  modes: ", nm)
    space = any(!has_tag_token(s, "Liouv") for s in bath.sites) ? "Hilbert" : "Liouville"
    println(io, "  space: ", space)
    site_dims = dim.(bath.sites)
    print(io, "  site dims: ")
    if length(site_dims) <= 10
        println(io, join(site_dims, ", "))
    else
        println(io, join(site_dims[1:5], ", "), ", ..., ", join(site_dims[(end - 4):end], ", "))
    end
    d_bath = isempty(bath.sites) ? 1 : prod(site_dims)
    println(io, "  bath Liouville dimension: ", d_bath)
    println(io)
    println(io, "  mode summary:")
    for k in (nm <= 6 ? (1:nm) : (1:2))
        m = bath.modes[k]
        println(io, "    [$k] BosonicMode(dim=$(dim(only(m.sites))), H_terms=$(length(terms(m.H))), coupling_terms=$(length(terms(m.coupling))), n_max=$(m.n_max))")
    end
    if nm > 6
        println(io, "    ⋮")
        for k in (nm - 1):nm
            m = bath.modes[k]
            println(io, "    [$k] BosonicMode(dim=$(dim(only(m.sites))), H_terms=$(length(terms(m.H))), coupling_terms=$(length(terms(m.coupling))), n_max=$(m.n_max))")
        end
    end
end

function Base.show(io::IO, bath::SpinBath)
    nm = length(bath.modes)
    println(io, "ProcessTensors.SpinBath")
    println(io, "  modes: ", nm)
    space = any(!has_tag_token(s, "Liouv") for s in bath.sites) ? "Hilbert" : "Liouville"
    println(io, "  space: ", space)
    site_dims = dim.(bath.sites)
    print(io, "  site dims: ")
    if length(site_dims) <= 10
        println(io, join(site_dims, ", "))
    else
        println(io, join(site_dims[1:5], ", "), ", ..., ", join(site_dims[(end - 4):end], ", "))
    end
    d_bath = isempty(bath.sites) ? 1 : prod(site_dims)
    println(io, "  bath Liouville dimension: ", d_bath)
    println(io)
    println(io, "  mode summary:")
    for k in (nm <= 6 ? (1:nm) : (1:2))
        m = bath.modes[k]
        println(io, "    [$k] SpinMode(dim=$(dim(only(m.sites))), H_terms=$(length(terms(m.H))), coupling_terms=$(length(terms(m.coupling))))")
    end
    if nm > 6
        println(io, "    ⋮")
        for k in (nm - 1):nm
            m = bath.modes[k]
            println(io, "    [$k] SpinMode(dim=$(dim(only(m.sites))), H_terms=$(length(terms(m.H))), coupling_terms=$(length(terms(m.coupling))))")
        end
    end
end

Base.show(io::IO, ::MIME"text/plain", mode::BosonicMode) = show(io, mode)
Base.show(io::IO, ::MIME"text/plain", mode::SpinMode) = show(io, mode)
Base.show(io::IO, ::MIME"text/plain", bath::BosonicBath) = show(io, bath)
Base.show(io::IO, ::MIME"text/plain", bath::SpinBath) = show(io, bath)

end # module Environments
