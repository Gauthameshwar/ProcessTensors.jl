
struct LiouvillianMPO{BasisTag} <: AbstractMPO
	mpo::Any
	metadata::NamedTuple
end

function build_liouvillian(mpo_hilbert, lindbladian_terms)
    # Follow row-major convention of the vectorization
    nothing
end

function density_to_liouville_mps(args...)
	nothing
end
