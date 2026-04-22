# This is a module that exports the classes of spectral functions that will be supported in this package. 

module SpectralFunctions

abstract type AbstractSpectralDensity end

struct OhmicSpectralDensity{T<:Real} <: AbstractSpectralDensity
    α::T    # Coupling strength
    ωc::T   # Cutoff frequency
    s::T    # s=1 (Ohmic), s>1 (Super-Ohmic), s<1 (Sub-Ohmic)
end

struct LorentzianSpectralDensity{T<:Real} <: AbstractSpectralDensity
    γ::T    # Coupling strength / width
    ω0::T   # Resonance frequency
end

# OhmicSpectral(; kwargs...) = OhmicSpectral((; kwargs...))
# PowerLawSpectral(; kwargs...) = PowerLawSpectral((; kwargs...))

end # module
