# src/Environments/environments.jl
module Environments

# Export the tools your users will need to build environments
export AbstractBathMode, BosonicMode, SpinMode, AbstractBath, BosonicBath, SpinBath

abstract type AbstractBathMode end

struct BosonicMode{T<:Real, O<:OpSum} <: AbstractBathMode
    initial_state::MPS{Hilbert}
    H::OpSum
    n_max::Int
end

struct SpinMode{T<:Real, O<:OpSum} <: AbstractBathMode
    initial_state::MPS{Hilbert}
    H::OpSum
end

abstract type AbstractBath end

struct BosonicBath{M <: AbstractBathMode, S <: AbstractSpectral}
    modes::Vector{M}
    spectral_density::S
    coupling::OpSum
end

struct SpinBath{M <: AbstractBathMode, S <: AbstractSpectral}
    modes::Vector{M}
    spectral_density::S
    coupling::OpSum
end


end # module
