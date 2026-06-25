# src/mps/observables.jl
# Tier C: observables forward to ITensorMPS on `.core` (see API page).

import ITensorMPS: inner, dot, ⋅, loginner, logdot, norm, lognorm,
                   expect, correlation_matrix, sample, sample!, entropy

inner(m1::AbstractMPS, m2::AbstractMPS) = inner(m1.core, m2.core)
inner(m1::AbstractMPS, A::AbstractMPS, m2::AbstractMPS) = inner(m1.core, A.core, m2.core)
dot(m1::AbstractMPS, m2::AbstractMPS) = dot(m1.core, m2.core)
loginner(m1::AbstractMPS, m2::AbstractMPS; kwargs...) = loginner(m1.core, m2.core; kwargs...)
logdot(m1::AbstractMPS, m2::AbstractMPS; kwargs...) = logdot(m1.core, m2.core; kwargs...)
norm(m::AbstractMPS) = norm(m.core)
lognorm(m::AbstractMPS) = lognorm(m.core)
expect(m::AbstractMPS, args...; kwargs...) = expect(m.core, args...; kwargs...)
correlation_matrix(m::AbstractMPS, args...; kwargs...) = correlation_matrix(m.core, args...; kwargs...)
sample(m::AbstractMPS) = sample(m.core)
sample!(m::AbstractMPS, args...; kwargs...) = sample!(m.core, args...; kwargs...)

import ITensors: svd, uniqueinds

# Entropy: orthogonalize at `b` on `.core` before the bond SVD (ITensorMPS convention).
function entropy(m::AbstractMPS, b::Integer)
    ITensorMPS.orthogonalize!(m.core, b)
    _, _, _, spec = svd(m.core[b], uniqueinds(m.core[b], m.core[b+1]))
    return entropy(spec)
end
