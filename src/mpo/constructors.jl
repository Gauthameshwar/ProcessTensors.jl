# src/mpo/constructors.jl

"""
    random_mpo(sites; kwargs...) -> MPO{Hilbert}

Construct a random Hilbert-space `MPO` by forwarding to `ITensorMPS.random_mpo`
and wrapping the returned core.

# Examples
```julia
s = siteinds("S=1/2", 4)
W = random_mpo(s)
```
"""
random_mpo(sites::Vector{<:Index}; kwargs...) = MPO{Hilbert}(ITensorMPS.random_mpo(sites; kwargs...))
