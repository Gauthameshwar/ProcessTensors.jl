# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: src/time_evolution/tdvp.jl
# Contributor: Gauthameshwar S.
#
# Provides TDVP helpers that forward to ITensorMPS and rewrap evolved states.

import ITensorMPS: tdvp

"""
    tdvp(H, t, psi::AbstractMPS; kwargs...) -> MPS

Run `ITensorMPS.tdvp` on `psi.core` and rewrap the result with the same
`Hilbert` or `Liouville` space tag as `psi`.

For algorithm details and keyword arguments, see the ITensorMPS documentation.
"""
tdvp(H, t::Number, psi::AbstractMPS; kwargs...) = _rewrap(psi, tdvp(H, t, psi.core; kwargs...))
tdvp(H::AbstractMPS, t::Number, psi::AbstractMPS; kwargs...) =
    _rewrap(psi, tdvp(H.core, t, psi.core; kwargs...))
