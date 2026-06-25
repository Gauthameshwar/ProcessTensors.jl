# src/time_evolution/tdvp.jl
# Tier C: TDVP helpers re-exported from ITensorMPS; `tdvp` rewraps the returned state (see API page).

import ITensorMPS: tdvp, promote_itensor_eltype, convert_leaf_eltype,
                   argsdict, sim!

"""
    tdvp(H, t, psi::AbstractMPS; kwargs...) -> MPS

Run `ITensorMPS.tdvp` on `psi.core` and rewrap the result with the same
`Hilbert` or `Liouville` space tag as `psi`.

For algorithm details and keyword arguments, see the ITensorMPS documentation.
"""
tdvp(H, t::Number, psi::AbstractMPS; kwargs...) = _rewrap(psi, tdvp(H, t, psi.core; kwargs...))
tdvp(H::AbstractMPS, t::Number, psi::AbstractMPS; kwargs...) =
    _rewrap(psi, tdvp(H.core, t, psi.core; kwargs...))
