# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: src/process_tensor/evaluate.jl
# Contributor: Gauthameshwar S.
#
# Implements process-tensor contraction and scalar/reduced-state evaluation
# algorithms.

import ITensorMPS: MPO as CoreMPO, MPS as CoreMPS
import ITensors: plev, scalar
import ITensors.Ops: Trotter

function _flatten_site_indices(sites::AbstractVector)
    idxs = Index[]
    for site in sites
        site isa Tuple ? append!(idxs, collect(site)) : push!(idxs, site)
    end
    return idxs
end

function _open_pt_leg_counts(idxs::AbstractVector{<:Index})
    n_input = count(idx -> plev(idx) == 1, idxs)
    n_output = count(idx -> plev(idx) == 0, idxs)
    return n_input, n_output
end

function _evaluate_result_shape_text(idxs::AbstractVector{<:Index})
    isempty(idxs) && return "scalar"
    return string(Tuple(dim.(idxs)))
end

function _evaluate_result_indices(result)
    if result isa ComplexF64
        return Index[]
    elseif result isa MPO{Liouville}
        return _flatten_site_indices(collect(siteinds(result)))
    elseif result isa ITensor
        return collect(inds(result))
    else
        throw(ArgumentError("evaluate_process: unexpected result type $(typeof(result))."))
    end
end

function _evaluate_result_summary(result)
    idxs = _evaluate_result_indices(result)
    n_input, n_output = _open_pt_leg_counts(idxs)
    if result isa ComplexF64
        result_type = "ComplexF64"
    elseif result isa MPO{Liouville}
        result_type = "MPO{Liouville}"
    else
        result_type = "ITensor"
    end
    return (
        open_input_sites=n_input,
        open_output_sites=n_output,
        result_type=_info_text(result_type),
        shape=_info_text(_evaluate_result_shape_text(idxs)),
    )
end

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
    info = open_leg_info(pt, seq)
    info.n_open_expected == 0 || return false
    isempty(info.missing_in) || return false
    isempty(info.missing_out) || return false
    final_instr = resolve_instrument(seq, pt.nsteps, seq.default)
    final_instr isa IdentityOperation && return false
    final_instr isa OpenOutput && return false
    return final_instr isa SingleLegInstrument
end

"""
    open_leg_info(pt, seq) -> NamedTuple

Report which process-tensor legs the schedule claims, which remain open, and
their dimensions.

Returns a named tuple with fields:

- `in_map`, `out_map`, `missing_in`, `missing_out` from [`instrument_leg_maps`](@ref)
- `open_in`, `open_out`: time labels owned by open bookkeeping instruments
- `n_open_expected`: expected number of uncontracted system legs after evaluation
- `open_dims`: dimensions of those expected open legs (from `pt`)
"""
function open_leg_info(pt::ProcessTensor, seq::InstrumentSeq)
    in_map, out_map, missing_in, missing_out = instrument_leg_maps(seq, pt.nsteps)
    default = seq.default

    open_in = Int[]
    open_out = Int[]
    for (tin, instr) in pairs(in_map)
        tin == 0 && continue
        if instr isa OpenInput || instr isa OpenInOut ||
           (instr isa ProductInstrument && instr.input_instr isa OpenInput)
            push!(open_in, tin)
        end
    end
    for (tout, instr) in pairs(out_map)
        if instr isa OpenOutput || instr isa OpenInOut ||
           (instr isa ProductInstrument && instr.output_instr isa OpenOutput)
            push!(open_out, tout)
        end
    end

    final_instr = resolve_instrument(seq, pt.nsteps, default)
    if final_instr isa OpenOutput || final_instr isa IdentityOperation
        push!(open_out, pt.nsteps - 1)
    end
    unique!(sort!(open_in))
    unique!(sort!(open_out))

    open_dims = Int[]
    for tin in open_in
        append!(open_dims, dim.(input_sites(pt, tin)))
    end
    for tout in open_out
        append!(open_dims, dim.(output_sites(pt, tout)))
    end

    return (
        in_map=in_map,
        out_map=out_map,
        missing_in=missing_in,
        missing_out=missing_out,
        open_in=open_in,
        open_out=open_out,
        n_open_expected=length(open_in) + length(open_out),
        open_dims=open_dims,
    )
end

"""
    evaluate_process(pt, seq; kwargs...) -> Union{ComplexF64, MPO{Liouville}, ITensor}

Contract a process tensor with an instrument schedule.

Return type depends on the number of uncontracted system legs after contraction:

| open legs | return |
|-----------|--------|
| 0 | `ComplexF64` |
| 1 | `MPO{Liouville}` |
| ≥ 2 | `ITensor` |

# Examples
```julia
seq = InstrumentSeq(default=IdentityOperation(), nsteps=pt.nsteps)
add!(seq, StatePreparation(ρ0), 0)
add!(seq, OpenOutput(), pt.nsteps)
result = evaluate_process(pt, seq)
```
"""
function _evaluate_process(
    pt::ProcessTensor,
    seq::InstrumentSeq;
    default_instr::AbstractInstrument=_schedule_default_instr(pt),
    alg=Trotter{2}(),
    all_legs_contracted::Union{Nothing,Bool}=nothing,
    reporter=nothing,
)
    _validate_instrument_schedule!(pt, seq, default_instr, "evaluate_process")

    # Keep "starting" through validation; callers may already have set that caption.
    _ensure_spinner!(reporter, "Evaluating process — starting")
    instruments = Instruments._create_instruments(
        pt,
        seq;
        default=default_instr,
        alg=alg,
        reporter=reporter,
        progress_desc="Evaluating process — materializing instruments",
    )
    _ensure_spinner!(reporter, "Evaluating process — preparing contraction")
    result = pt.core[1] * instruments[1]
    contract = function (update!)
        for step in 1:(pt.nsteps - 1)
            result *= instruments[step + 1]
            result *= pt.core[step + 1]
            update!(step)
        end
    end
    if reporter === nothing
        contract(_ -> nothing)
    else
        _with_progress(
            contract,
            reporter,
            "Evaluating process — contracting",
            max(pt.nsteps - 1, 1),
        )
    end
    result *= instruments[pt.nsteps + 1]

    n_open = length(inds(result))

    legs_closed = something(all_legs_contracted, all_pt_legs_contracted(pt, seq))
    
    # a small type-stability issue here with the return being a Union{ComplexF64, MPO{Liouville}, ITensor}
    if legs_closed
        n_open == 0 || throw(
            ArgumentError(
                "evaluate_process: expected 0 uncontracted indices " *
                "(all_pt_legs_contracted=true) but found $n_open.",
            ),
        )
        return ComplexF64(scalar(result))
    elseif n_open == 1
        keep = Index[only(inds(result))]
        rho_liouv = _liouville_mps_from_itensor(result, keep)
        return MPO{Liouville}(CoreMPO(collect(rho_liouv.core)), rho_liouv.combiners)
    else
        return result
    end
end

function evaluate_process(
    pt::ProcessTensor,
    seq::InstrumentSeq;
    default_instr::AbstractInstrument=_schedule_default_instr(pt),
    alg=Trotter{2}(),
    all_legs_contracted::Union{Nothing,Bool}=nothing,
    progress::Union{Bool,Symbol}=:auto,
    verbose::Bool=false,
)
    reporter = _progress_reporter(progress)
    started = time()
    try
        _ensure_spinner!(reporter, "Evaluating process — starting")
        result = _evaluate_process(
            pt,
            seq;
            default_instr=default_instr,
            alg=alg,
            all_legs_contracted=all_legs_contracted,
            reporter=reporter,
        )
        _clear_stage!(reporter)
        if verbose
            summary = _evaluate_result_summary(result)
            @info "Evaluated process" nsteps=pt.nsteps elapsed_seconds=(time() - started) summary...
        end
        return result
    finally
        _clear_stage!(reporter)
    end
end

"""
    evaluate_process(pt, seqs::AbstractVector{<:InstrumentSeq}; kwargs...) -> Vector{ComplexF64}

Evaluate a batch of fully contracted instrument schedules and return one scalar
per schedule.
"""
function evaluate_process(
    pt::ProcessTensor,
    seqs::AbstractVector{<:InstrumentSeq};
    default_instr::AbstractInstrument=_schedule_default_instr(pt),
    alg=Trotter{2}(),
    progress::Union{Bool,Symbol}=:auto,
    verbose::Bool=false,
)
    results = Vector{ComplexF64}(undef, length(seqs))
    reporter = _progress_reporter(progress)
    started = time()
    evaluate_batch = function (update!)
        for i in eachindex(seqs)
            val = _evaluate_process(
                pt,
                seqs[i];
                default_instr=default_instr,
                alg=alg,
                reporter=nothing,
            )
            val isa ComplexF64 || throw(
                ArgumentError(
                    "evaluate_process(batch): schedule at index $i is not fully contracted; " *
                    "batch overload requires scalar schedules (all_pt_legs_contracted=true).",
                ),
            )
            results[i] = val
            update!(i)
        end
    end
    try
        _ensure_spinner!(reporter, "Evaluating process batch — starting")
        _with_progress(evaluate_batch, reporter, "Evaluating process batch — schedules", length(seqs))
        _clear_stage!(reporter)
    finally
        _clear_stage!(reporter)
    end
    verbose && @info "Evaluated process batch" schedules=length(seqs) nsteps=pt.nsteps open_input_sites=0 open_output_sites=0 result_type=_info_text("ComplexF64") shape=_info_text("scalar") elapsed_seconds=(time() - started)
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
