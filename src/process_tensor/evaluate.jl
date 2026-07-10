# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: src/process_tensor/evaluate.jl
# Contributor: Gauthameshwar S.
#
# Implements process-tensor contraction and scalar/reduced-state evaluation
# algorithms.

import ITensorMPS: MPO as CoreMPO, MPS as CoreMPS
import ITensors: scalar
import ITensors.Ops: Trotter

# Wrap a contracted ITensor (reduced density-matrix vector) into an MPS{Liouville}.
function _liouville_mps_from_itensor(t::ITensor, liouv_sites::AbstractVector{<:Index})
    length(liouv_sites) == 1 || throw(ArgumentError("_liouville_mps_from_itensor currently supports a single Liouville site."))
    liouv_site = only(liouv_sites)
    t_loc = t
    if !hasind(t_loc, liouv_site)
        hasind(t_loc, prime(liouv_site)) || throw(
            ArgumentError("_liouville_mps_from_itensor: reduced tensor is missing Liouville site $(liouv_site)."),
        )
        t_loc = replaceind(t_loc, prime(liouv_site), liouv_site)
    end
    phys_site = _phys_site_from_liouv(liouv_site)
    d = dim(phys_site)
    d2 = dim(liouv_site)
    d * d == d2 || throw(ArgumentError("_liouville_mps_from_itensor expects dim(liouv_site)=d^2."))

    # Convert PT's local Liouville basis ordering to the package's canonical ordering
    # used by to_liouville/to_hilbert by applying vec(ρ) -> vec(ρᵀ).
    perm = zeros(ComplexF64, d2, d2)
    for i in 1:d, j in 1:d
        old = (i - 1) * d + j
        new = (j - 1) * d + i
        perm[new, old] = 1.0
    end
    transpose_map = ITensor(perm, prime(liouv_site), liouv_site)
    t_loc = transpose_map * t_loc
    t_loc = replaceind(t_loc, prime(liouv_site), liouv_site)

    comb = combiner(phys_site, prime(phys_site); tags=tags(liouv_site))
    comb = replaceind(comb, combinedind(comb), liouv_site)
    return MPS{Liouville}(CoreMPS([t_loc]), ITensor[comb])
end

"""
    all_pt_legs_contracted(pt, seq) -> Bool

Return `true` when the schedule closes every process-tensor leg, so
[`evaluate_process`](@ref) returns a `ComplexF64` scalar.
"""
function all_pt_legs_contracted(pt::ProcessTensor, seq::InstrumentSeq)
    _, _, missing_in, missing_out = instrument_leg_maps(seq, pt.nsteps)
    isempty(missing_in) || return false
    isempty(missing_out) || return false
    final_instr = resolve_instrument(seq, pt.nsteps, seq.default)
    # Terminal Identity is treated as OpenOutput by create_instruments.
    final_instr isa IdentityOperation && return false
    final_instr isa OpenOutput && return false
    return final_instr isa SingleLegInstrument
end

"""
    evaluate_process(pt, seq; kwargs...) -> Union{ComplexF64, MPO{Liouville}}

Contract a process tensor with an instrument schedule.

Return a scalar when every process-tensor leg is closed, or an
`MPO{Liouville}` when one system output leg is left open by [`OpenOutput`](@ref)
(or by omitting a terminal closer).

# Examples
```julia
seq = InstrumentSeq(default=IdentityOperation(), nsteps=pt.nsteps)
add!(seq, StatePreparation(ρ0), 0)
add!(seq, OpenOutput(), pt.nsteps)
result = evaluate_process(pt, seq)
```
"""
function evaluate_process(
    pt::ProcessTensor,
    seq::InstrumentSeq;
    default_instr::AbstractInstrument=_schedule_default_instr(pt),
    alg=Trotter{2}(),
    all_legs_contracted::Union{Nothing,Bool}=nothing,
)
    _validate_instrument_schedule!(pt, seq, default_instr, "evaluate_process")

    instruments = create_instruments(pt, seq; default=default_instr, alg=alg)
    legs_closed = something(all_legs_contracted, all_pt_legs_contracted(pt, seq))

    result = pt.core[1] * instruments[1]
    for step in 1:(pt.nsteps - 1)
        result *= instruments[step + 1]
        result *= pt.core[step + 1]
    end
    result *= instruments[pt.nsteps + 1]

    if legs_closed
        n_open = length(inds(result))
        n_open == 0 || throw(
            ArgumentError(
                "evaluate_process: expected 0 uncontracted indices " *
                "(all_pt_legs_contracted=true) but found $n_open.",
            ),
        )
        return ComplexF64(scalar(result))
    end

    n_open = length(inds(result))
    n_open == 1 || throw(
        ArgumentError(
            "evaluate_process: expected exactly one open system leg, found $n_open.",
        ),
    )
    keep = Index[only(inds(result))]
    rho_liouv = _liouville_mps_from_itensor(result, keep)
    return MPO{Liouville}(CoreMPO(collect(rho_liouv.core)), rho_liouv.combiners)
end

"""
    evaluate_process(pt, seqs::AbstractVector{<:InstrumentSeq}; kwargs...) -> Vector{ComplexF64}

Evaluate a batch of fully contracted instrument schedules and return one scalar
per schedule.
"""
function evaluate_process(
    pt::ProcessTensor,
    seqs::AbstractVector{<:InstrumentSeq};
    kwargs...
)
    results = Vector{ComplexF64}(undef, length(seqs))
    for i in eachindex(seqs)
        val = evaluate_process(pt, seqs[i]; kwargs...)
        val isa ComplexF64 || throw(
            ArgumentError(
                "evaluate_process(batch): schedule at index $i is not fully contracted; " *
                "batch overload requires scalar schedules (all_pt_legs_contracted=true).",
            ),
        )
        results[i] = val
    end
    return results
end

"""
    evaluate_process(pt, rho0, seq; kwargs...)

Insert `StatePreparation(rho0)` at `tstep = 0` and contract the resulting
schedule with `pt`.
"""
function evaluate_process(
    pt::ProcessTensor,
    rho0,
    seq::InstrumentSeq;
    default_instr::AbstractInstrument=_schedule_default_instr(pt),
    kwargs...
)
    seq_full = InstrumentSeq(seq.default, seq.nsteps; entries=Dict{Int,AbstractInstrument}(pairs(seq.entries)))
    add!(seq_full, StatePreparation(rho0), 0)
    return evaluate_process(pt, seq_full; default_instr=default_instr, kwargs...)
end

"""
    evaluate_process(pt, rho0; kwargs...)

Evaluate a process tensor from an initial state using the default instrument
schedule.
"""
function evaluate_process(
    pt::ProcessTensor,
    rho0;
    default_instr::AbstractInstrument=_schedule_default_instr(pt),
    kwargs...
)
    seq = InstrumentSeq(default=default_instr, nsteps=pt.nsteps)
    add!(seq, StatePreparation(rho0), 0)
    return evaluate_process(pt, seq; default_instr=default_instr, kwargs...)
end
