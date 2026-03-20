module ProcessTensors

using ITensors
import ITensorMPS

# =========================================================================
# The ProcessTensors Public API
# =========================================================================

include("basis.jl")

using .Basis: AbstractSpace, Hilbert, Liouville

include("mps/mps.jl")
include("mps/mps_utils.jl")
include("mpo/mpo.jl")
include("mpo/mpo_utils.jl")

import ITensorMPS: siteinds, siteind, linkinds, linkind, linkdim, linkdims, maxlinkdim,
                   common_siteind, common_siteinds, unique_siteind, unique_siteinds,
                   findfirstsiteind, findfirstsiteinds, findsite, findsites, firstsiteind, firstsiteinds,
                   replace_siteinds, replace_siteinds!, totalqn, hassameinds,
                   isortho, ortho_lims, orthocenter, set_ortho_lims!, reset_ortho_lims!, @preserve_ortho,
                   error_contract, truncerror, truncerrors,
                   inner, dot, ⋅, loginner, logdot, norm, lognorm, expect, correlation_matrix,
                   sample, sample!, apply, contract, add, truncate!, truncate,
                   orthogonalize!, orthogonalize, normalize!, replacebond, replacebond!, swapbondsites, movesite, movesites,
                   OpSum, add!, op, ops, random_mps, random_mpo, state, outer, projector,
                   tdvp, TimeDependentSum, Trotter, promote_itensor_eltype, convert_leaf_eltype,
                   argsdict, sim!, splitblocks, tr, entropy, eigs, to_vec, coefficient, replaceprime

# We re-export tools that users will need from ITensors / ITensorMPS
export siteinds, siteind, linkinds, linkind, linkdim, linkdims, maxlinkdim,
       common_siteind, common_siteinds, unique_siteind, unique_siteinds,
       findfirstsiteind, findfirstsiteinds, findsite, findsites, firstsiteind, firstsiteinds,
       replace_siteinds, replace_siteinds!, totalqn, hassameinds,
       isortho, ortho_lims, orthocenter, set_ortho_lims!, reset_ortho_lims!, @preserve_ortho,
       error_contract, truncerror, truncerrors,
       inner, dot, ⋅, loginner, logdot, norm, lognorm, expect, correlation_matrix,
       sample, sample!, apply, contract, add, truncate!, truncate,
       orthogonalize!, orthogonalize, normalize!, replacebond, replacebond!, swapbondsites, movesite, movesites,
       OpSum, add!, op, ops, random_mps, random_mpo, state, outer, projector,
       tdvp, TimeDependentSum, Trotter, promote_itensor_eltype, convert_leaf_eltype,
       argsdict, sim!, splitblocks, tr, entropy, eigs, to_vec, coefficient, replaceprime

end
