# src/networks/manipulations.jl

import ITensorMPS: replacebond, replacebond!, swapbondsites, movesite, movesites

# Out-of-place (returns new MPS/MPO)
replacebond(m::AbstractMPS, b::Integer, phi::ITensor; kwargs...) = _rewrap(m, replacebond(m.core, b, phi; kwargs...))

swapbondsites(m::AbstractMPS, b::Integer; kwargs...) = _rewrap(m, swapbondsites(m.core, b; kwargs...))

movesite(m::AbstractMPS, args...; kwargs...) = _rewrap(m, movesite(m.core, args...; kwargs...))
movesite(m::AbstractMPS, n1n2::Pair{Int, Int}; kwargs...) = _rewrap(m, movesite(m.core, n1n2; kwargs...))

movesites(m::AbstractMPS, args...; kwargs...) = _rewrap(m, movesites(m.core, args...; kwargs...))
movesites(m::AbstractMPS, nsns::Vector{Pair{Int, Int}}; kwargs...) = _rewrap(m, movesites(m.core, nsns; kwargs...))

# In-place mutating
replacebond!(m::AbstractMPS, b::Integer, phi::ITensor; kwargs...) = (replacebond!(m.core, b, phi; kwargs...); m)
