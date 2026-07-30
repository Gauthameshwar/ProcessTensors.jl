# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: src/networks/algebra.jl
# Contributor: Gauthameshwar S.
#
# Provides compact network algebra helpers that forward to ITensorMPS and rewrap
# Hilbert/Liouville results.

import ITensorMPS: apply, contract, add

apply(op::AbstractMPS, m::AbstractMPS; kwargs...) =
    _rewrap(m, apply(op.core, m.core; kwargs...))
apply(op, m::AbstractMPS; kwargs...) =
    _rewrap(m, apply(op, m.core; kwargs...))
apply(op::ITensor, m::AbstractMPS; kwargs...) =
    _rewrap(m, apply(op, m.core; kwargs...))
apply(op::Vector{ITensor}, m::AbstractMPS; kwargs...) =
    _rewrap(m, apply(op, m.core; kwargs...))
apply(op::ITensors.LazyApply.Prod{ITensor}, m::AbstractMPS; kwargs...) =
    _rewrap(m, apply(op, m.core; kwargs...))

contract(op::AbstractMPS, m::AbstractMPS; kwargs...) =
    _rewrap(m, contract(op.core, m.core; kwargs...))
contract(op, m::AbstractMPS; kwargs...) =
    _rewrap(m, contract(op, m.core; kwargs...))

add(op::AbstractMPS, m::AbstractMPS; kwargs...) =
    _rewrap(m, add(op.core, m.core; kwargs...))
add(op, m::AbstractMPS; kwargs...) =
    _rewrap(m, add(op, m.core; kwargs...))
add(m1::CoreAbstractMPS, m2::AbstractMPS; kwargs...) =
    _rewrap(m2, add(m1, m2.core; kwargs...))

# Arithmetic on wrappers extends Base
import Base: +, -, *
+(m1::AbstractMPS, m2::AbstractMPS; kwargs...) =
    _rewrap(m1, +(m1.core, m2.core; kwargs...))
-(m1::AbstractMPS, m2::AbstractMPS; kwargs...) =
    _rewrap(m1, -(m1.core, m2.core; kwargs...))
*(c::Number, m::AbstractMPS) = _rewrap(m, c * m.core)
*(m::AbstractMPS, c::Number) = _rewrap(m, m.core * c)
