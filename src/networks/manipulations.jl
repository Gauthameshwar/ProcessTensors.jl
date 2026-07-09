# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: src/networks/manipulations.jl
# Contributor: Gauthameshwar S.
#
# Provides bond and site manipulation helpers that forward to ITensorMPS and rewrap.

import ITensorMPS: replacebond, replacebond!, swapbondsites, movesite, movesites

replacebond(m::AbstractMPS, b::Integer, phi::ITensor; kwargs...) = _rewrap(m, replacebond(m.core, b, phi; kwargs...))
swapbondsites(m::AbstractMPS, b::Integer; kwargs...) = _rewrap(m, swapbondsites(m.core, b; kwargs...))
movesite(m::AbstractMPS, n1n2::Pair{Int, Int}; kwargs...) = _rewrap(m, movesite(m.core, n1n2; kwargs...))
movesites(m::AbstractMPS, nsns::Vector{Pair{Int, Int}}; kwargs...) = _rewrap(m, movesites(m.core, nsns; kwargs...))
replacebond!(m::AbstractMPS, b::Integer, phi::ITensor; kwargs...) = (replacebond!(m.core, b, phi; kwargs...); m)
