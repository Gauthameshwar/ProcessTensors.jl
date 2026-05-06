# src/mps/observables.jl

import ITensorMPS: inner, dot, ⋅, loginner, logdot, norm, lognorm,
                   expect, correlation_matrix, sample, sample!, entropy

# Two-MPS-arg observables (return scalars)
# Note: ⋅ is an alias for dot, so no separate forwarding needed
inner(m1::AbstractMPS, m2::AbstractMPS) = inner(m1.core, m2.core)
inner(m1::AbstractMPS, A::AbstractMPS, m2::AbstractMPS) = inner(m1.core, A.core, m2.core)
dot(m1::AbstractMPS, m2::AbstractMPS) = dot(m1.core, m2.core)
loginner(m1::AbstractMPS, m2::AbstractMPS; kwargs...) = loginner(m1.core, m2.core; kwargs...)
logdot(m1::AbstractMPS, m2::AbstractMPS; kwargs...) = logdot(m1.core, m2.core; kwargs...)

# Single-MPS-arg observables (return scalars / vectors)
norm(m::AbstractMPS) = norm(m.core)
lognorm(m::AbstractMPS) = lognorm(m.core)
expect(m::AbstractMPS, args...; kwargs...) = expect(m.core, args...; kwargs...)
correlation_matrix(m::AbstractMPS, args...; kwargs...) = correlation_matrix(m.core, args...; kwargs...)

import ITensors: svd, uniqueinds
# Entropy of a bond (computes von Neumann entropy of the Singular Values)
function entropy(m::AbstractMPS, b::Integer)
    ITensorMPS.orthogonalize!(m.core, b)
    _, _, _, spec = svd(m.core[b], uniqueinds(m.core[b], m.core[b+1]))
    return entropy(spec)
end

# Sampling
sample(m::AbstractMPS) = sample(m.core)
sample!(m::AbstractMPS, args...; kwargs...) = sample!(m.core, args...; kwargs...)