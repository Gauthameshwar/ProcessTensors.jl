module CoreContracts

abstract type AbstractSpace end
struct Hilbert <: AbstractSpace end
struct Liouville <: AbstractSpace end

abstract type AbstractBasis end
struct SpinBasis <: AbstractBasis end
struct FermionBasis <: AbstractBasis end
struct BosonBasis <: AbstractBasis end
struct CustomBasis{T} <: AbstractBasis end

abstract type AbstractBathParticle end
struct Boson <: AbstractBathParticle end
struct Fermion <: AbstractBathParticle end
struct Spin <: AbstractBathParticle end

validate_space(::Type{<:AbstractSpace})
validate_basis(::Type{<:AbstractBasis})
validate_dimensions(args...)

compatible(::Type{<:AbstractSpace}, ::Type{<:AbstractSpace}) = true
compatible(::Type{<:AbstractBasis}, ::Type{<:AbstractBasis}) = true

end # module CoreContracts
