# src/mpo/manipulations.jl

import ITensorMPS: splitblocks

# Out-of-place (returns new MPO)
splitblocks(m::AbstractMPS; kwargs...) = _rewrap(m, splitblocks(m.core; kwargs...))