# src/networks/manipulations.jl
# Tier C: bond/site manipulations forward to ITensorMPS and rewrap (see API page).

import ITensorMPS: replacebond, replacebond!, swapbondsites, movesite, movesites

replacebond(m::AbstractMPS, b::Integer, phi::ITensor; kwargs...) = _rewrap(m, replacebond(m.core, b, phi; kwargs...))
swapbondsites(m::AbstractMPS, b::Integer; kwargs...) = _rewrap(m, swapbondsites(m.core, b; kwargs...))
movesite(m::AbstractMPS, n1n2::Pair{Int, Int}; kwargs...) = _rewrap(m, movesite(m.core, n1n2; kwargs...))
movesites(m::AbstractMPS, nsns::Vector{Pair{Int, Int}}; kwargs...) = _rewrap(m, movesites(m.core, nsns; kwargs...))
replacebond!(m::AbstractMPS, b::Integer, phi::ITensor; kwargs...) = (replacebond!(m.core, b, phi; kwargs...); m)
