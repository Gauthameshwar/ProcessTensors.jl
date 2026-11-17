# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: src/networks/observables.jl
# Contributor: Gauthameshwar S.
#
# Provides compact MPS/MPO observable helpers that forward to ITensorMPS on
# wrapped cores.

import ITensorMPS: inner, dot, norm, expect, correlation_matrix, tr

inner(m1::AbstractMPS, m2::AbstractMPS; kwargs...) =
    inner(m1.core, m2.core; kwargs...)
inner(m1::AbstractMPS, A::AbstractMPS, m2::AbstractMPS; kwargs...) =
    inner(m1.core, A.core, m2.core; kwargs...)
dot(m1::AbstractMPS, m2::AbstractMPS; kwargs...) =
    dot(m1.core, m2.core; kwargs...)
norm(m::AbstractMPS; kwargs...) = norm(m.core; kwargs...)
expect(m::AbstractMPS, args...; kwargs...) =
    expect(m.core, args...; kwargs...)
correlation_matrix(m::AbstractMPS, args...; kwargs...) =
    correlation_matrix(m.core, args...; kwargs...)
tr(m::AbstractMPS; kwargs...) = tr(m.core; kwargs...)
