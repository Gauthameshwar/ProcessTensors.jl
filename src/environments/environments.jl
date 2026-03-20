# src/Environments/environments.jl
module Environments

# Export the tools your users will need to build environments
export Bath, Fermion, Boson, Spin, BathMode

abstract type AbstractParticle end

# Singleton types for particles with fixed properties
struct Fermion <: AbstractParticle end
struct Boson <: AbstractParticle end
struct Spin{N} <: AbstractParticle end

struct BathMode{P<:AbstractBathParticle} <: AbstractBathMode
    initial_state::Any
    metadata::NamedTuple
end

# We parameterize this so the compiler generates specialized, fast machine code 
# for every specific combination of particle and spectral density.
struct Bath{P <: AbstractParticle, S <: AbstractSpectral}
    modes::Vector{BathMode{P}}
    spectral_function::S
    coupling_operator::String  
end

end
