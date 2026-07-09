# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: src/networks/orthogonality.jl
# Contributor: Gauthameshwar S.
#
# Provides orthogonality helpers that forward to ITensorMPS on wrapped cores.

import ITensorMPS: isortho, ortho_lims, orthocenter, set_ortho_lims!, reset_ortho_lims!,
                   orthogonalize!, orthogonalize, normalize!, @preserve_ortho

isortho(m::AbstractMPS) = isortho(m.core)
ortho_lims(m::AbstractMPS) = ortho_lims(m.core)
orthocenter(m::AbstractMPS) = orthocenter(m.core)
orthogonalize(m::AbstractMPS, j::Int; kwargs...) = _rewrap(m, orthogonalize(m.core, j; kwargs...))
orthogonalize!(m::AbstractMPS, j::Int; kwargs...) = (ITensorMPS.orthogonalize!(m.core, j; kwargs...); m)
normalize!(m::AbstractMPS) = (ITensorMPS.normalize!(m.core); m)
set_ortho_lims!(m::AbstractMPS, args...) = (ITensorMPS.set_ortho_lims!(m.core, args...); m)
set_ortho_lims!(m::AbstractMPS, r::UnitRange{Int}) = (ITensorMPS.set_ortho_lims!(m.core, r); m)
reset_ortho_lims!(m::AbstractMPS) = (ITensorMPS.reset_ortho_lims!(m.core); m)
