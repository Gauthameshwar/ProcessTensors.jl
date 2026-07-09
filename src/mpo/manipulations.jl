# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: src/mpo/manipulations.jl
# Contributor: Gauthameshwar S.
#
# Provides MPO manipulation helpers that forward to ITensorMPS and rewrap.

import ITensorMPS: splitblocks

splitblocks(m::AbstractMPS; kwargs...) = _rewrap(m, splitblocks(linkinds, m.core; kwargs...))
splitblocks(::typeof(linkinds), m::AbstractMPS; kwargs...) = _rewrap(m, splitblocks(linkinds, m.core; kwargs...))
