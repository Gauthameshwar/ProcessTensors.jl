# src/networks/algebra.jl

import ITensorMPS: apply, contract, add, truncate!, truncate, error_contract, truncerror, truncerrors

# Query functions (return scalars)
truncerror(m::AbstractMPS) = truncerror(m.core)
truncerrors(m::AbstractMPS) = truncerrors(m.core)

error_contract(m1::AbstractMPS, m2::AbstractMPS; kwargs...) = error_contract(m1.core, m2.core; kwargs...)

# Out-of-place (returns new MPS/MPO)
truncate(m::AbstractMPS; kwargs...) = _rewrap(m, truncate(m.core; kwargs...))

# In-place mutating
truncate!(m::AbstractMPS; kwargs...) = (truncate!(m.core; kwargs...); m)

# Two-operand operations (operator × state → new state)
for func in (:apply, :contract, :add)
    @eval begin
        $func(op::AbstractMPS, m::AbstractMPS; kwargs...) = _rewrap(m, $func(op.core, m.core; kwargs...))
        $func(op, m::AbstractMPS; kwargs...) = _rewrap(m, $func(op, m.core; kwargs...))
    end
end

import Base: +, -, *
+(m1::AbstractMPS, m2::AbstractMPS; kwargs...) = _rewrap(m1, +(m1.core, m2.core; kwargs...))
-(m1::AbstractMPS, m2::AbstractMPS; kwargs...) = _rewrap(m1, -(m1.core, m2.core; kwargs...))
*(c::Number, m::AbstractMPS) = _rewrap(m, c * m.core)
*(m::AbstractMPS, c::Number) = _rewrap(m, m.core * c)
