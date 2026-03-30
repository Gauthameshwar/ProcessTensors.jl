# src/mpo/constructors.jl

# random_mpo: wrap ITensorMPS.random_mpo result into ProcessTensors.MPO
random_mpo(sites::Vector{<:Index}; kwargs...) = MPO{Hilbert}(ITensorMPS.random_mpo(sites; kwargs...))
random_mpo(eltype::Type{<:Number}, sites::Vector{<:Index}; kwargs...) = MPO{Hilbert}(ITensorMPS.random_mpo(eltype, sites; kwargs...))