# src/mpo/manipulations.jl
# Tier C: `splitblocks` forwards to ITensorMPS and rewraps (see API page).

import ITensorMPS: splitblocks

splitblocks(m::AbstractMPS; kwargs...) = _rewrap(m, splitblocks(linkinds, m.core; kwargs...))
splitblocks(::typeof(linkinds), m::AbstractMPS; kwargs...) = _rewrap(m, splitblocks(linkinds, m.core; kwargs...))
