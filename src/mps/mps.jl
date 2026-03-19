# This module exports the basic classes of MPS used in the package

module mps

abstract type AbstractMPS end

struct MPS{SpaceTag,BasisTag} <: AbstractMPS
	tensors::Any
	metadata::NamedTuple
end

MPS(tensors; kwargs...) = MPS{Any,Any}(tensors, (; kwargs...))
MPS(::Type{S}, ::Type{B}, tensors; kwargs...) where {S,B} = MPS{S,B}(tensors, (; kwargs...))

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

function validate_mps(args...)
	nothing
end

end # module
