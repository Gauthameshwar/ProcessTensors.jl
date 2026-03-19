module Hamiltonians

struct HamiltonianMPO{BasisTag} <: AbstractMPO
	mpo::Any
	metadata::NamedTuple
end

function build_term(args...) # Hamiltonian block terms that can consist of on-site or two-site terms
    nothing
end

function coupling_term(args...) # ITensor entity that stores the coupling between the system and bath with an ITensor that only contains the index of the system operator, the bath operator, and the data. 
    nothing
end

function build_hamiltonian(args...)
    nothing
end

function ishermitian_hamiltonian(args...)
    nothing
end

end # module Hamiltonians
