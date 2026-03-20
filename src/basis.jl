# src/basis.jl
module Basis

abstract type AbstractSpace end
struct Hilbert <: AbstractSpace end
struct Liouville <: AbstractSpace end

end # module Basis
