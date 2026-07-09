# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: src/mpo/constructors.jl
# Contributor: Gauthameshwar S.
#
# Provides MPO constructor helpers that forward to ITensorMPS and wrap results.

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
