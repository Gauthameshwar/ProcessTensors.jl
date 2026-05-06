# src/environments/spectrals.jl

module Spectrals

export AbstractSpectralDensity, OhmicSpectralDensity, LorentzianSpectralDensity,
       ohmic_sd, lorentzian_sd

abstract type AbstractSpectralDensity end

struct OhmicSpectralDensity{T<:Real} <: AbstractSpectralDensity
    alpha::T
    wc::T
    s::T
end

struct LorentzianSpectralDensity{T<:Real} <: AbstractSpectralDensity
    lambda::T
    gamma::T
    omega0::T
end

ohmic_sd(; alpha::Real=1.0, wc::Real=1.0, s::Real=1.0) =
    OhmicSpectralDensity(float(alpha), float(wc), float(s))

lorentzian_sd(; lambda::Real=1.0, gamma::Real=1.0, omega0::Real=0.0) =
    LorentzianSpectralDensity(float(lambda), float(gamma), float(omega0))

end # module Spectrals
