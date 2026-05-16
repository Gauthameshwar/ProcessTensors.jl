# src/time_evolution/tdvp.jl

import ITensorMPS: tdvp, promote_itensor_eltype, convert_leaf_eltype,
                   argsdict, sim!

# Forward tdvp for wrapped MPS/MPO (ITensorMPS requires explicit time)
tdvp(H, t::Number, psi::AbstractMPS; kwargs...) = _rewrap(psi, tdvp(H, t, psi.core; kwargs...))
tdvp(H::AbstractMPS, t::Number, psi::AbstractMPS; kwargs...) =
    _rewrap(psi, tdvp(H.core, t, psi.core; kwargs...))