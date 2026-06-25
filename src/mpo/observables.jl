# src/mpo/observables.jl
# Tier C: trace forwards to ITensorMPS on `.core` (see API page).

import ITensorMPS: tr

tr(m::AbstractMPS; kwargs...) = tr(m.core; kwargs...)
