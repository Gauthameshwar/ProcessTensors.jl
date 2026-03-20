# src/mps/mps_utils.jl

# Import the functions from ITensorMPS that we want to extend
import ITensorMPS: truncate, orthogonalize, replacebond, replace_siteinds, swapbondsites, movesite, movesites
import ITensorMPS: apply, contract, add
import ITensorMPS: truncate!, orthogonalize!, normalize!, replacebond!, replace_siteinds!
import ITensorMPS: random_mps, outer, projector

# Helper to rewrap ITensor objects back into our AbstractMPS container
function _rewrap(m::AbstractMPS{S}, new_core) where {S<:AbstractSpace}
    if m isa MPS
        return S === Hilbert ? MPS{Hilbert}(new_core) : MPS{Liouvillian}(new_core, m.combiners)
    else
        return S === Hilbert ? MPO{Hilbert}(new_core) : MPO{Liouvillian}(new_core, m.combiners)
    end
end

# Out-of-place functions (returning a new MPS/MPO)
for func in (:truncate, :orthogonalize, :replacebond, :replace_siteinds, :swapbondsites, :movesite, :movesites)
    @eval begin
        $func(m::AbstractMPS, args...; kwargs...) = _rewrap(m, $func(m.core, args...; kwargs...))
    end
end

# Two-argument operations: apply, contract, add
for func in (:apply, :contract, :add)
    @eval begin
        $func(op::AbstractMPS, m::AbstractMPS; kwargs...) = _rewrap(m, $func(op.core, m.core; kwargs...))
        $func(op, m::AbstractMPS; kwargs...) = _rewrap(m, $func(op, m.core; kwargs...))
    end
end

# Generators and constructor-like functions
random_mps(args...; kwargs...) = MPS(ITensorMPS.random_mps(args...; kwargs...))

projector(m::AbstractMPS; kwargs...) = MPO(projector(m.core; kwargs...))
outer(m1::AbstractMPS, m2::AbstractMPS; kwargs...) = MPO(outer(m1.core, m2.core; kwargs...))
