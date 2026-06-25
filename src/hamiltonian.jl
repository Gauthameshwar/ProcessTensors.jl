# src/hamiltonian.jl
# Tier C: `OpSum` utilities are re-exported from ITensorMPS (see API page).

import ITensorMPS: OpSum, add!, op, ops, eigs, coefficient

const MAX_DENSE_LIOUVILLE_DIM = 5_000

function _validate_dense_liouville_budget(d_joint::Integer; context::AbstractString)
    d_joint <= MAX_DENSE_LIOUVILLE_DIM && return nothing
    @warn "$context: joint Liouville vector dimension D=$d_joint exceeds MAX_DENSE_LIOUVILLE_DIM=$(MAX_DENSE_LIOUVILLE_DIM)."
    throw(
        ArgumentError(
            "$context: joint Liouville vector dimension D=$d_joint is too large for dense exp(dt * L). " *
            "Please reduce mode count / local cutoff or wait for TEBD-based large-bath support.",
        ),
    )
end
