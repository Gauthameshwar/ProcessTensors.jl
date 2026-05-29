# src/environments/spectrals.jl

"""
    ProcessTensors.Spectrals

Spectral-density helpers for future bath models (e.g. ACE / structured environments).
They are **not** re-exported from `ProcessTensors`; use `ProcessTensors.Spectrals` or
`using ProcessTensors.Spectrals` in calling code. As of current releases, built process
tensors do **not** fold these functions into the joint Liouvillian; defaults like
`ohmic_sd()` on baths are for API consistency and forward compatibility.
"""
module Spectrals

export AbstractSpectralDensity, OhmicSpectralDensity, LorentzianSpectralDensity,
       ohmic_sd, lorentzian_sd

abstract type AbstractSpectralDensity end

"""Ohmic spectral density parameters; reserved for future ACE/TEMPO coupling — not used in `build_process_tensor` yet."""
struct OhmicSpectralDensity{T<:Real} <: AbstractSpectralDensity
    alpha::T
    wc::T
    s::T
end

"""Lorentzian spectral density parameters; reserved for future ACE/TEMPO coupling — not used in `build_process_tensor` yet."""
struct LorentzianSpectralDensity{T<:Real} <: AbstractSpectralDensity
    lambda::T
    gamma::T
    omega0::T
end

ohmic_sd(; alpha::Real=1.0, wc::Real=1.0, s::Real=1.0) =
    OhmicSpectralDensity(float(alpha), float(wc), float(s))

lorentzian_sd(; lambda::Real=1.0, gamma::Real=1.0, omega0::Real=0.0) =
    LorentzianSpectralDensity(float(lambda), float(gamma), float(omega0))

function Base.show(io::IO, sd::OhmicSpectralDensity)
    print(io, "Ohmic(α=", sd.alpha, ", ωc=", sd.wc, ", s=", sd.s, ")")
end

function Base.show(io::IO, sd::LorentzianSpectralDensity)
    print(io, "Lorentzian(λ=", sd.lambda, ", γ=", sd.gamma, ", ω₀=", sd.omega0, ")")
end

Base.show(io::IO, ::MIME"text/plain", sd::OhmicSpectralDensity) = show(io, sd)
Base.show(io::IO, ::MIME"text/plain", sd::LorentzianSpectralDensity) = show(io, sd)

end # module Spectrals
