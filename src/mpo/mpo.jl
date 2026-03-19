# This module exports the basic classes of MPO that will be used in this package

module mpo

abstract type AbstractMPO end

struct MPO{SpaceTag,BasisTag} <: AbstractMPO
	tensors::Any
	metadata::NamedTuple
end

MPO(tensors; kwargs...) = MPO{Any,Any}(tensors, (; kwargs...))
MPO(::Type{S}, ::Type{B}, tensors; kwargs...) where {S,B} = MPO{S,B}(tensors, (; kwargs...))

function bondinds(args...)
	nothing
end

function siteinds(args...)
	nothing
end

function basis(args...)
	nothing
end

function space(args...)
	nothing
end

function validate_mpo(args...)
	nothing
end

end # module
