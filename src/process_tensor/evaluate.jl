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

# Traces all indices except `keep`; for multimode PT this includes fused bath-memory links.
function _trace_out_except(t::ITensor, keep::AbstractVector{<:Index}; k::Int=0, environment=nothing)
    function _is_fused_bath_link(idx::Index, environment)
        environment isa AbstractBath || return false
        length(environment.modes) > 1 || return false
        "Link" in tag_tokens(idx) || return false
        return dim(idx) == prod(dim(only(mode.sites)) for mode in environment.modes)
    end
    
    function _fused_bath_trace_itensor(environment::AbstractBath, fused_link::Index, k::Int)
        bath_sites_prime = prime.([only(mode.sites) for mode in environment.modes])
        comb_primed = combiner(bath_sites_prime...; tags="PT,Link,FusedBath,Prime")
        bath_trace = ITensor(1.0)
        for site in bath_sites_prime
            bath_trace *= instrument_itensor(TraceOut(; leg_plev=plev(site)), Index[site], k)
        end
        return replaceind(bath_trace * comb_primed, combinedind(comb_primed), fused_link)
    end
    
    keep_vec = Index[keep...]
    out = t
    for idx in inds(out)
        idx in keep_vec && continue
        tstep_tag = tag_value(idx, "tstep=")
        idx_k = isnothing(tstep_tag) ? k : parse(Int, tstep_tag)
        trace_tensor = if _is_fused_bath_link(idx, environment)
            _fused_bath_trace_itensor(environment, idx, idx_k)
        else
            instrument_itensor(TraceOut(; leg_plev=plev(idx)), Index[idx], idx_k)
        end
        out *= trace_tensor
    end
    return out
end

# Wrap a contracted ITensor core (reduced density matrix vector) into an MPS{Liouville} object
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

function _open_output_steps(seq::InstrumentSeq, nsteps::Int, default::AbstractInstrument)
    cuts = Int[]
    for step in 1:(nsteps - 1)
        resolve_instrument(seq, step, default) isa OpenOutput && push!(cuts, step)
    end
    return cuts
end

"""
    all_pt_legs_contracted(pt, seq) -> Bool

Return `true` when the schedule closes every process-tensor leg, so
[`evaluate_process`](@ref) returns a `ComplexF64` scalar.
"""
function all_pt_legs_contracted(pt::ProcessTensor, seq::InstrumentSeq)
    !isempty(_open_output_steps(seq, pt.nsteps, seq.default)) && return false
    _, _, missing_in, missing_out = instrument_leg_maps(seq, pt.nsteps)
    isempty(missing_in) || return false
    isempty(missing_out) || return false
    final_instr = resolve_instrument(seq, pt.nsteps, seq.default)
    return final_instr isa TraceOut || final_instr isa SingleLegInstrument
end

"""
    evaluate_process(pt, seq; kwargs...) -> Union{ComplexF64, MPO{Liouville}}

Contract a process tensor with an instrument schedule.

Return a scalar when every process-tensor leg is closed, or an
`MPO{Liouville}` when one system output leg is left open.

# Examples
```julia
seq = InstrumentSeq(default=IdentityOperation(), nsteps=pt.nsteps)
add!(seq, StatePreparation(ρ0), 0)
add!(seq, TraceOut(), pt.nsteps)
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

    open_cuts = _open_output_steps(seq, pt.nsteps, default_instr)
    open_keep_k = if isempty(open_cuts)
        nothing
    else
        length(open_cuts) == 1 || throw(
            ArgumentError(
                "evaluate_process: expected at most one OpenOutput in the schedule; found at steps $(open_cuts).",
            ),
        )
        open_cuts[1] - 1
    end
    instruments = create_instruments(pt, seq; default=default_instr, alg=alg)
    legs_closed = something(all_legs_contracted, all_pt_legs_contracted(pt, seq))

    result = pt.core[1] * instruments[1]
    for step in 1:(pt.nsteps - 1)
        instr = resolve_instrument(seq, step, default_instr)
        out_prev, in_curr = coupling_times(pt, step)
        if instr isa OpenOutput
            tmp = copy(pt.core[step + 1])
            tmp *= instrument_itensor(instr, in_curr, out_prev, step; dt=pt.dt, alg=alg)
            result *= tmp
        else
            result *= instruments[step + 1]
            result *= pt.core[step + 1]
        end
    end

    final_instr = resolve_instrument(seq, pt.nsteps, seq.default)
    if final_instr isa TraceOut || final_instr isa SingleLegInstrument
        out_prev, _ = coupling_times(pt, pt.nsteps)
        result *= instrument_itensor(final_instr, out_prev, pt.nsteps - 1)
    end

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

    keep_k = something(open_keep_k, pt.nsteps - 1)
    keep = output_sites(pt, keep_k)
    reduced = if open_keep_k === nothing && length(inds(result)) == 1
        result
    else
        _trace_out_except(result, keep; k=keep_k, environment=pt.environment)
    end
    n_open = length(inds(reduced))
    n_open == 1 || throw(
        ArgumentError(
            "evaluate_process: expected exactly one open system leg at k=$keep_k, found $n_open.",
        ),
    )
    rho_liouv = _liouville_mps_from_itensor(reduced, keep)
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
