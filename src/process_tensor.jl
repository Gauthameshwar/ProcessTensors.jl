# This module defines the ProcessTensorMPO type and associated functions for working with process tensors in the MPO representation.

struct ProcessTensorMPO{BasisTag} <: AbstractMPO
	mpo::Any
	metadata::NamedTuple
end

function process_tensor_mpo(args...)
	nothing
end

function validate_pt(args...)
    nothing
end
