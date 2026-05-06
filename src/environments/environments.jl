# src/environments/environments.jl

module Environments

using ..ProcessTensors: AbstractMPS, OpSum, Index, siteinds
using ..Spectrals: AbstractSpectralDensity, ohmic_sd
using ITensors: dim

export AbstractBathMode, AbstractBath, BosonicMode, SpinMode, BosonicBath, SpinBath,
       bosonic_mode, spin_mode, bosonic_bath, spin_bath,
       mode_initial_states

abstract type AbstractBathMode end
abstract type AbstractBath end

##########  Bath Particle Modes  ##########
# Bosonic Bath Mode
struct BosonicMode{M<:AbstractMPS} <: AbstractBathMode
    rho0::M # must have liouville index space
    H::OpSum
    n_max::Int
    sites::Vector{Index} # must be in liouville space

    # Constructor to verify the indices input, n_max, and H are all consistent
    function BosonicMode{M}(sites::AbstractVector{<:Index}, H::OpSum, n_max::Int, rho0::M) where {M<:AbstractMPS}
        length(sites) == 1 || throw(ArgumentError("BosonicMode: a single bosonic mode should have exactly one site index. Got $(length(sites))."))
        siteinds(rho0) == sites || throw(ArgumentError("BosonicMode:rho0 and sites must have the same indices. Got $(siteinds(rho0)) and $(sites)."))
        n_max == dim(only(sites)) - 1 || throw(ArgumentError("BosonicMode:n_max must be dim(sites) - 1. Got $n_max for sites with dim=$(dim(only(sites)))."))
        H == OpSum() && @warn "BosonicMode:H is empty. This is usually not what you want."
        new(rho0, H, n_max, Index[sites...])
    end
end

###### Convenience functions for constructing the BosonicMode on a user-level #######
# BosonicMode([Index1], OpSum(), 5, MPS{Liouville})
function BosonicMode(sites::AbstractVector{<:Index}, H::OpSum, n_max::Int, rho0::M) where {M<:AbstractMPS}
    return BosonicMode{M}(sites, H, n_max, rho0)
end
# BosonicMode([Index1], OpSum(), MPS{Liouville}; n_max=5)
BosonicMode(sites::AbstractVector{<:Index}, H::OpSum, rho0::AbstractMPS; n_max::Int=dim(only(sites)) - 1) =
    BosonicMode(sites, H, n_max, rho0)
# BosonicMode(sites=[Index1], H=OpSum(), rho0=MPS{Liouville}, n_max=5)
BosonicMode(; sites::AbstractVector{<:Index}, H::OpSum=OpSum(), rho0::AbstractMPS, n_max::Int=dim(only(sites)) - 1) =
    BosonicMode(sites, H, n_max, rho0)

# Spin Bath Mode
struct SpinMode{M<:AbstractMPS} <: AbstractBathMode
    rho0::M # must have liouville index space
    H::OpSum
    sites::Vector{Index} # must be in liouville space

    # Constructor to verify the indices input and H are all consistent
    function SpinMode{M}(sites::AbstractVector{<:Index}, H::OpSum, rho0::M) where {M<:AbstractMPS}
        length(sites) == 1 || throw(ArgumentError("SpinMode: a single spin mode should have exactly one site index. Got $(length(sites))."))
        siteinds(rho0) == sites || throw(ArgumentError("SpinMode:rho0 and sites must have the same indices. Got $(siteinds(rho0)) and $(sites)."))
        H == OpSum() && @warn "SpinMode:H is empty. This is usually not what you want."
        new(rho0, H, Index[sites...])
    end
end

###### Convenience functions for constructing the SpinMode on a user-level #######
# SpinMode([Index1], OpSum(), MPS{Liouville})
function SpinMode(sites::AbstractVector{<:Index}, H::OpSum, rho0::AbstractMPS)
    return SpinMode{typeof(rho0)}(sites, H, rho0)
end
# SpinMode(sites=[Index1], H=OpSum(), rho0=MPS{Liouville})
SpinMode(; sites::AbstractVector{<:Index}, H::OpSum=OpSum(), rho0::AbstractMPS) =
    SpinMode(sites, H, rho0)

##########  Bath Objects  ##########
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
        coupling == OpSum() && @warn "BosonicBath: system-bath coupling is empty. This is usually not what you want."
        new(collect(modes), spectral_density, coupling, Index[sites...])
    end
end

###### Convenience functions for constructing the BosonicBath on a user-level #######
# BosonicBath([Index1, Index2], [BosonicMode1, BosonicMode2, ...], ohmic_sd(), OpSum())
function BosonicBath(
    sites::AbstractVector{<:Index},
    modes::AbstractVector{M},
    spectral_density::S,
    coupling::O,
) where {M<:BosonicMode,S<:AbstractSpectralDensity,O<:OpSum}
    return BosonicBath{M,S,O}(sites, modes, spectral_density, coupling)
end
# BosonicBath([BosonicMode1, BosonicMode2, ...], ohmic_sd(), OpSum())
function BosonicBath(
    modes::AbstractVector{<:BosonicMode},
    spectral_density::AbstractSpectralDensity=ohmic_sd(),
    coupling::OpSum=OpSum(),
)
    sites = collect(Iterators.flatten(getfield.(modes, :sites)))
    return BosonicBath(sites, modes, spectral_density, coupling)
end
# BosonicBath(modes=[BosonicMode1, BosonicMode2, ...], spectral_density=ohmic_sd(), coupling=OpSum())
function BosonicBath(; modes::AbstractVector=BosonicMode[], spectral_density::AbstractSpectralDensity=ohmic_sd(), coupling::OpSum=OpSum())
    all(mode -> mode isa BosonicMode, modes) || throw(ArgumentError("BosonicBath:modes must contain only BosonicMode values."))
    return BosonicBath(collect(BosonicMode, modes), spectral_density, coupling)
end

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
        coupling == OpSum() && @warn "SpinBath: system-bath coupling is empty. This is usually not what you want."
        new(collect(modes), spectral_density, coupling, Index[sites...])
    end
end

###### Convenience functions for constructing the SpinBath on a user-level #######
# SpinBath([Index1, Index2], [SpinMode1, SpinMode2, ...], ohmic_sd(), OpSum())
function SpinBath(
    sites::AbstractVector{<:Index},
    modes::AbstractVector{M},
    spectral_density::S,
    coupling::O,
) where {M<:SpinMode,S<:AbstractSpectralDensity,O<:OpSum}
    return SpinBath{M,S,O}(sites, modes, spectral_density, coupling)
end
# SpinBath([SpinMode1, SpinMode2, ...], ohmic_sd(), OpSum())
function SpinBath(
    modes::AbstractVector{<:SpinMode},
    spectral_density::AbstractSpectralDensity=ohmic_sd(),
    coupling::OpSum=OpSum(),
)
    sites = collect(Iterators.flatten(getfield.(modes, :sites)))
    return SpinBath(sites, modes, spectral_density, coupling)
end
# SpinBath(modes=[SpinMode1, SpinMode2, ...], spectral_density=ohmic_sd(), coupling=OpSum())
function SpinBath(; modes::AbstractVector=SpinMode[], spectral_density::AbstractSpectralDensity=ohmic_sd(), coupling::OpSum=OpSum())
    all(mode -> mode isa SpinMode, modes) || throw(ArgumentError("SpinBath:modes must contain only SpinMode values."))
    return SpinBath(collect(SpinMode, modes), spectral_density, coupling)
end

# Allow users to call all the above functions without the caps
bosonic_mode(args...; kwargs...) = BosonicMode(args...; kwargs...)
spin_mode(args...; kwargs...) =
    SpinMode(args...; kwargs...)

bosonic_bath(modes::AbstractVector{<:BosonicMode}; spectral_density::AbstractSpectralDensity=ohmic_sd(), coupling::OpSum=OpSum()) =
    BosonicBath(modes, spectral_density, coupling)
bosonic_bath(; modes::AbstractVector=BosonicMode[], spectral_density::AbstractSpectralDensity=ohmic_sd(), coupling::OpSum=OpSum()) =
    BosonicBath(; modes=modes, spectral_density=spectral_density, coupling=coupling)

spin_bath(modes::AbstractVector{<:SpinMode}; spectral_density::AbstractSpectralDensity=ohmic_sd(), coupling::OpSum=OpSum()) =
    SpinBath(modes, spectral_density, coupling)
spin_bath(; modes::AbstractVector=SpinMode[], spectral_density::AbstractSpectralDensity=ohmic_sd(), coupling::OpSum=OpSum()) =
    SpinBath(; modes=modes, spectral_density=spectral_density, coupling=coupling)

mode_initial_states(bath::AbstractBath) = getfield.(bath.modes, :rho0)

end # module Environments
