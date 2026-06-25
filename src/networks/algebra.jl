# src/networks/algebra.jl
# Tier C: algebra forwards to ITensorMPS on `.core` and rewraps (see API page).

import ITensors: product, ITensor
import ITensorMPS: apply, contract, add, truncate!, truncate, error_contract

_core_or_self(x) = x isa AbstractMPS ? x.core : x
error_contract(args...; kwargs...) = error_contract((_core_or_self.(args))...; kwargs...)
truncate(m::AbstractMPS; kwargs...) = _rewrap(m, truncate(m.core; kwargs...))
truncate!(m::AbstractMPS; kwargs...) = (truncate!(m.core; kwargs...); m)

for func in (:apply, :contract, :add)
    @eval begin
        $func(op::AbstractMPS, m::AbstractMPS; kwargs...) = _rewrap(m, $func(op.core, m.core; kwargs...))
        $func(op, m::AbstractMPS; kwargs...) = _rewrap(m, $func(op, m.core; kwargs...))
    end
end

add(m1::CoreAbstractMPS, m2::AbstractMPS; kwargs...) = _rewrap(m2, add(m1, m2.core; kwargs...))
apply(op::ITensor, m::AbstractMPS; kwargs...) = _rewrap(m, apply(op, m.core; kwargs...))
apply(op::Vector{ITensor}, m::AbstractMPS; kwargs...) = _rewrap(m, apply(op, m.core; kwargs...))
apply(op::ITensors.LazyApply.Prod{ITensor}, m::AbstractMPS; kwargs...) = _rewrap(m, apply(op, m.core; kwargs...))

import Base: +, -, *
+(m1::AbstractMPS, m2::AbstractMPS; kwargs...) = _rewrap(m1, +(m1.core, m2.core; kwargs...))
-(m1::AbstractMPS, m2::AbstractMPS; kwargs...) = _rewrap(m1, -(m1.core, m2.core; kwargs...))
*(c::Number, m::AbstractMPS) = _rewrap(m, c * m.core)
*(m::AbstractMPS, c::Number) = _rewrap(m, m.core * c)
