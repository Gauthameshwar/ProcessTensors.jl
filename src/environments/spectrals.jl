# This is a module that exports the classes of spectral functions that will be supported in this package. 

module SpectralFunctions

abstract type AbstractSpectral end

struct OhmicSpectral <: AbstractSpectral
	params::NamedTuple
end

struct PowerLawSpectral <: AbstractSpectral
	params::NamedTuple
end

struct CustomSpectral{F} <: AbstractSpectral
	f::F
	params::NamedTuple
end

OhmicSpectral(; kwargs...) = OhmicSpectral((; kwargs...))
PowerLawSpectral(; kwargs...) = PowerLawSpectral((; kwargs...))

function spectral_density(args...)
	nothing
end

function validate_spectral_model(args...)
	nothing
end

end # module
