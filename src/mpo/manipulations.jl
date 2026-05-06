# src/mpo/manipulations.jl

import ITensorMPS: splitblocks

# Out-of-place (returns new MPO)
splitblocks(m::AbstractMPS; kwargs...) = _rewrap(m, splitblocks(linkinds, m.core; kwargs...))
splitblocks(::typeof(linkinds), m::AbstractMPS; kwargs...) = _rewrap(m, splitblocks(linkinds, m.core; kwargs...))