# src/basis.jl
module Basis

"""
    AbstractSpace

Type tag interface for the tensor-network space carried by ProcessTensors MPS/MPO wrappers.
"""
abstract type AbstractSpace end

"""
    Hilbert

Type tag for ordinary Hilbert-space tensor networks.

`MPS{Hilbert}` represents a state vector, and `MPO{Hilbert}` represents an operator
or density matrix on physical site indices.
"""
struct Hilbert <: AbstractSpace end

"""
    Liouville

Type tag for vectorized density matrices and superoperators.

Liouville-space site indices have local dimension ``d^2`` for a Hilbert-space site
of dimension ``d``. They are used by [`to_liouville`](@ref ProcessTensors.to_liouville),
[`to_hilbert`](@ref ProcessTensors.to_hilbert),
Liouvillian MPOs, instruments, and process-tensor contractions.
"""
struct Liouville <: AbstractSpace end

end # module Basis
