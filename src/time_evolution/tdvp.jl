# src/time_evolution/tdvp.jl

import ITensorMPS: tdvp, promote_itensor_eltype, convert_leaf_eltype,
                   argsdict, sim!

# Forward tdvp for wrapped MPS/MPO
tdvp(H, psi::AbstractMPS; kwargs...) = _rewrap(psi, tdvp(H, psi.core; kwargs...))
tdvp(H::AbstractMPS, psi::AbstractMPS; kwargs...) = _rewrap(psi, tdvp(H.core, psi.core; kwargs...))
tdvp(H, t::Number, psi::AbstractMPS; kwargs...) = _rewrap(psi, tdvp(H, t, psi.core; kwargs...))
tdvp(H::AbstractMPS, t::Number, psi::AbstractMPS; kwargs...) =
    _rewrap(psi, tdvp(H.core, t, psi.core; kwargs...))