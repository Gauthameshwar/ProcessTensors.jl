module ProcessTensors

using ITensors
import ITensorMPS

# =========================================================================
# Foundation
# =========================================================================

include("basis.jl")
using .Basis: AbstractSpace, Hilbert, Liouville

# =========================================================================
# Core Types (MPS & MPO structs, getproperty, show)
# =========================================================================

include("mps/mps.jl")
include("mpo/mpo.jl")

# Rewrap utility — converts a raw CoreMPS/CoreMPO back into our wrapper type
function _rewrap(m::AbstractMPS{S}, new_core) where {S <: AbstractSpace}
    if m isa MPS
        return S === Hilbert ? MPS{Hilbert}(new_core) : MPS{Liouville}(new_core, m.combiners)
    else
        return S === Hilbert ? MPO{Hilbert}(new_core) : MPO{Liouville}(new_core, m.combiners)
    end
end

# =========================================================================
# Network Operations
# =========================================================================

include("networks/indices.jl")
include("networks/algebra.jl")
include("networks/manipulations.jl")
include("networks/orthogonality.jl")

# =========================================================================
# MPS-Specific
# =========================================================================

include("mps/constructors.jl")
include("mps/observables.jl")

# =========================================================================
# MPO-Specific
# =========================================================================

include("mpo/constructors.jl")
include("mpo/manipulations.jl")
include("mpo/observables.jl")

# =========================================================================
# Hamiltonian / Operator Sums
# =========================================================================

include("hamiltonian.jl")

# =========================================================================
# Liouvillian
# =========================================================================

include("liouvillian.jl")

# =========================================================================
# Time Evolution
# =========================================================================

include("time_evolution/tdvp.jl")
include("time_evolution/tebd.jl")

# =========================================================================
# ProcessTensors.jl module: Systems / Baths / Instruments
# =========================================================================

include("systems/systems.jl")
include("environments/spectrals.jl")
using .Spectrals: AbstractSpectralDensity
include("environments/environments.jl")
using .Environments: AbstractBathMode, AbstractBath, BosonicMode, SpinMode, BosonicBath, SpinBath,
                    bosonic_mode, spin_mode, bosonic_bath, spin_bath,
                    mode_initial_states
include("systems/instruments.jl")
using .Instruments: AbstractInstrument, SingleLegInstrument, TwoLegInstrument,
                    StatePreparation, ObservableMeasurement, TraceOut,
                    IdentityOperation, SystemPropagation, resolve_instrument,
                    InstrumentSeq, add!, instrument_itensor, instrument_leg_maps

# =========================================================================
# Process Tensors
# =========================================================================

include("process_tensor.jl")

# =========================================================================
# Exports (grouped by category)
# =========================================================================

# Core types
export AbstractMPS, AbstractMPO, MPS, MPO, AbstractSpace, Hilbert, Liouville
export tag_tokens, has_tag_token, has_tag_prefix, tag_value

# Network: indices
export siteinds, siteind, linkinds, linkind, linkdim, linkdims, maxlinkdim,
       common_siteind, common_siteinds, unique_siteind, unique_siteinds,
       findfirstsiteind, findfirstsiteinds, findsite, findsites,
       firstsiteind, firstsiteinds,
       replace_siteinds, replace_siteinds!, hassameinds, totalqn, replaceprime

# Network: algebra
export apply, contract, add, truncate!, truncate,
       error_contract, truncerror, truncerrors

# Network: manipulations
export replacebond, replacebond!, swapbondsites, movesite, movesites

# Network: orthogonality
export isortho, ortho_lims, orthocenter, set_ortho_lims!, reset_ortho_lims!,
       orthogonalize!, orthogonalize, normalize!, @preserve_ortho

# MPS constructors & observables
export random_mps, state, outer, projector,
       inner, dot, ⋅, loginner, logdot, norm, lognorm,
       expect, correlation_matrix, sample, sample!, entropy

# MPO constructors, manipulations & observables
export random_mpo, splitblocks, tr

# Hamiltonian / OpSum
export OpSum, add!, op, ops, eigs, coefficient

# Liouvillian
export to_dm, to_liouville, to_hilbert, liouv_sites, MPO_Liouville, OpSum_Liouville

# Systems / Baths / Instruments / PT
export AbstractSystem, SpinSystem, BosonSystem, spin_system, boson_system

export AbstractBathMode, BosonicMode, SpinMode, bosonic_mode, spin_mode,
       AbstractBath, BosonicBath, SpinBath, bosonic_bath, spin_bath,
       mode_initial_states

export AbstractInstrument, SingleLegInstrument, TwoLegInstrument,
       StatePreparation, ObservableMeasurement, TraceOut,
       IdentityOperation, SystemPropagation, resolve_instrument,
       InstrumentSeq, add!, instrument_itensor, instrument_leg_maps

export ProcessTensor, build_process_tensor, default_schedule, evolve,
       coupling_times, coupling_sites, input_sites, output_sites,
       create_instruments, generate_pt_legs

# Time evolution
export tdvp, tebd, Trotter,
       promote_itensor_eltype, convert_leaf_eltype, argsdict, sim!

end # module ProcessTensors
