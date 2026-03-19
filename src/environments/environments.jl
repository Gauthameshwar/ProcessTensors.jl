# This module exports the classes that will be used to represent the environment in this package

module Environments

abstract type AbstractBathMode end

struct BathMode{P<:AbstractBathParticle} <: AbstractBathMode
    initial_state::Any
    metadata::NamedTuple
end

const BosonMode = BathMode{Boson}
const FermionMode = BathMode{Fermion}
const SpinMode = BathMode{Spin}

BathMode(::Type{P}; kwargs...) where {P<:AbstractBathParticle} = BathMode{P}((; kwargs...))

struct Environment
    modes::Any
    spectral_func::Any
    coupling::Any
    metadata::NamedTuple
end

Environment(; kwargs...) = Environment(nothing, nothing, nothing, nothing, nothing, (; kwargs...))

function add_mode(args...)
    nothing
end

function remove_mode(args...)
    nothing
end

function set_spectral_model(args...)
    nothing
end

function thermal_mode(args...)
    nothing
end

function set_coupling(args...)
    nothing
end

function validate_environment(args...)
    nothing
end

end # module
