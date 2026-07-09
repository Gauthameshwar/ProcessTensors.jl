# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: src/mpo/observables.jl
# Contributor: Gauthameshwar S.
#
# Provides MPO observable helpers that forward to ITensorMPS on wrapped cores.

import ITensorMPS: tr

tr(m::AbstractMPS; kwargs...) = tr(m.core; kwargs...)
