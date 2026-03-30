# src/mpo/observables.jl

import ITensorMPS: tr

# Trace (returns scalar)
tr(m::AbstractMPS; kwargs...) = tr(m.core; kwargs...)