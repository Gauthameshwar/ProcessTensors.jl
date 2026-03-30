# src/networks/orthogonality.jl

import ITensorMPS: isortho, ortho_lims, orthocenter, set_ortho_lims!, reset_ortho_lims!,
                   orthogonalize!, orthogonalize, normalize!, @preserve_ortho

# Query functions (return booleans / ranges / integers)
isortho(m::AbstractMPS) = isortho(m.core)
ortho_lims(m::AbstractMPS) = ortho_lims(m.core)
orthocenter(m::AbstractMPS) = orthocenter(m.core)

# Out-of-place (returns new MPS/MPO)
orthogonalize(m::AbstractMPS, j::Int; kwargs...) = _rewrap(m, orthogonalize(m.core, j; kwargs...))

# In-place mutating — use Int (== Int64) to beat ITensorMPS's Int64 signature
orthogonalize!(m::AbstractMPS, j::Int; kwargs...) = (ITensorMPS.orthogonalize!(m.core, j; kwargs...); m)
normalize!(m::AbstractMPS) = (ITensorMPS.normalize!(m.core); m)
set_ortho_lims!(m::AbstractMPS, args...) = (ITensorMPS.set_ortho_lims!(m.core, args...); m)
set_ortho_lims!(m::AbstractMPS, r::UnitRange{Int}) = (ITensorMPS.set_ortho_lims!(m.core, r); m)
reset_ortho_lims!(m::AbstractMPS) = (ITensorMPS.reset_ortho_lims!(m.core); m)
