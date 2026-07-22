# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: src/instruments/itensor_instruments.jl
# Contributor: Gauthameshwar S.
#
# Materializes lazy instruments and schedules as ITensor objects on
# process-tensor legs.

# Contract all cores of a wrapped MPS/MPO into one dense ITensor for local PT-leg insertion.
_mps_to_itensor(state::AbstractMPS) = foldl(*, state)
_mpo_to_itensor(op_mpo::AbstractMPO) = foldl(*, op_mpo)

# Replace source indices by the target PT indices while preserving tensor values.
function _reindex_itensor(t::ITensor, old_sites::AbstractVector{<:Index}, new_sites::AbstractVector{<:Index})
    length(old_sites) == length(new_sites) || throw(ArgumentError("Cannot reindex ITensor: site count mismatch."))
    tout = t
    for (old_s, new_s) in zip(old_sites, new_sites)
        old_s == new_s && continue
        tout = replaceind(tout, old_s, new_s)
    end
    return tout
end

# `TraceOut` contracts a Liouville leg with the vectorized Hilbert-space identity.
function _vectorized_identity_itensor(pt_sites::AbstractVector{<:Index})
    vecI = ITensor(1.0)
    for liouville_site in pt_sites
        d2 = dim(liouville_site)
        d = isqrt(d2)
        d * d == d2 || throw(ArgumentError("TraceOut requires Liouville-site dimensions d^2; got dim=$d2."))
        s = Index(d, "site")
        sprime = prime(s)
        deltaId = delta(s, sprime)
        cmb = combiner(s, sprime)
        Ivec = deltaId * cmb
        Ivec = replaceind(Ivec, combinedind(cmb), liouville_site)
        vecI *= Ivec
    end
    return vecI
end

# Instrument materialization always targets Liouville PT sites.
_coerce_liouville_state(rho0::AbstractMPS{Liouville}, sites::AbstractVector{<:Index}) =
    rho0
_coerce_liouville_state(rho0::AbstractMPO{Hilbert}, sites::AbstractVector{<:Index}) =
    to_liouville(rho0; sites=sites)
_coerce_liouville_state(rho0::AbstractMPS{Hilbert}, sites::AbstractVector{<:Index}) =
    to_liouville(to_dm(rho0); sites=sites)

_hilbert_density_from_prep(ρ::AbstractMPO{Hilbert}) = ρ
_hilbert_density_from_prep(ψ::AbstractMPS{Hilbert}) = to_dm(ψ)
function _hilbert_density_from_prep(state)
    throw(ArgumentError("StatePreparation state must be Hilbert MPS or MPO."))
end

function _composed_hilbert_mpo(instr::_ComposedSingleLegInstrument, sites::AbstractVector{<:Index})
    factors = instr.factors

    prep_idx = findfirst(f -> f isa StatePreparation, factors)
    ρ = prep_idx === nothing ? nothing : _hilbert_density_from_prep(factors[prep_idx].state)

    phys_sites = if ρ === nothing
        Index[_phys_site_from_liouv(s) for s in sites]
    else
        _phys_sites_from_hilbert_mpo(ρ)
    end

    op_acc = if any(f -> f isa ObservableMeasurement, factors)
        _fold_observable_factors(factors, phys_sites)
    else
        nothing
    end

    hilbert_mpo = if ρ === nothing && op_acc === nothing
        nothing
    elseif ρ === nothing
        op_acc
    elseif op_acc === nothing
        ρ
    elseif first(factors) isa StatePreparation
        apply(ρ, op_acc)
    else
        apply(op_acc, ρ)
    end

    hilbert_mpo === nothing && throw(
        ArgumentError("_ComposedSingleLegInstrument: no factors to build."),
    )

    return hilbert_mpo
end

"""
    instrument_itensor(instr, pt_sites, k; kwargs...)
    instrument_itensor(instr, input_pt_sites, output_pt_sites, k; kwargs...)

Materialize an instrument as an `ITensor` on process-tensor legs.

Hilbert-space preparations and observables are converted to Liouville-space
objects when needed.
"""
function instrument_itensor(
    instr::StatePreparation,
    pt_sites_arg::AbstractVector{<:Index},
    k::Int;
    kwargs...,
)
    sites = isempty(instr.pt_sites) ? Index[pt_sites_arg...] : instr.pt_sites
    _validate_single_leg_sites("StatePreparation", sites, instr.leg_plev)
    all(s -> _tstep_from_site(s) in (nothing, k), sites) || throw(
        ArgumentError("StatePreparation: all pt_sites must have tstep=$k when tagged."),
    )
    stateL = _coerce_liouville_state(instr.state, sites)
    state_t = _mps_to_itensor(stateL)
    return _reindex_itensor(state_t, siteinds(stateL), sites)
end

function instrument_itensor(
    instr::_ComposedSingleLegInstrument,
    pt_sites_arg::AbstractVector{<:Index},
    k::Int;
    kwargs...,
)
    sites = isempty(instr.pt_sites) ? Index[pt_sites_arg...] : instr.pt_sites
    _validate_single_leg_sites("_ComposedSingleLegInstrument", sites, instr.leg_plev)
    all(s -> _tstep_from_site(s) in (nothing, k), sites) || throw(
        ArgumentError("_ComposedSingleLegInstrument: all pt_sites must have tstep=$k when tagged."),
    )
    hilbert_mpo = _composed_hilbert_mpo(instr, sites)
    state_l = to_liouville(hilbert_mpo; sites=sites)
    return _reindex_itensor(_mps_to_itensor(state_l), siteinds(state_l), sites)
end

function instrument_itensor(
    instr::LeftRightOperator,
    input_pt_sites::AbstractVector{<:Index},
    output_pt_sites::AbstractVector{<:Index},
    k::Int;
    kwargs...,
)
    in_sites = isempty(instr.input_pt_sites) ? Index[input_pt_sites...] :
instr.input_pt_sites
    out_sites = isempty(instr.output_pt_sites) ? Index[output_pt_sites...] :
instr.output_pt_sites
    _validate_two_leg_map("LeftRightOperator", in_sites, out_sites)

    inp = only(in_sites)
    out = only(out_sites)

    # Extract physical sites and convert MPOs to matrices
    phys_sites = _phys_sites_from_hilbert_mpo(instr.left)
    _phys_sites_from_hilbert_mpo(instr.right) == phys_sites || throw(
        ArgumentError("LeftRightOperator: left and right MPOs must share physical sites."),
    )

    T_A = _mpo_to_itensor(instr.left)
    T_B = _mpo_to_itensor(instr.right)
    d = prod(dim.(phys_sites))

    # Extract as matrices
    A_mat = reshape(ComplexF64.(Array(T_A, prime.(phys_sites)..., phys_sites...)), d, d)
    B_mat = reshape(ComplexF64.(Array(T_B, prime.(phys_sites)..., phys_sites...)), d, d)

    # Use existing _superop_matrix helpers to build Liouville embeddings
    Id = Matrix{ComplexF64}(I, d, d)
    left_superop = _superop_matrix(_LiouvLeft(), A_mat, Id)    # I ⊗ A
    right_superop = _superop_matrix(_LiouvRight(), B_mat, Id)  # B^T ⊗ I

    # Compose superoperators: (B^T ⊗ I) * (I ⊗ A) = B^T ⊗ A
    W = right_superop * left_superop

    # Return ITensor with BOTH indices properly bound
    return ITensor(reshape(W, d^2, d^2), inp, out)
end

function instrument_itensor(
    instr::ObservableMeasurement,
    pt_sites_arg::AbstractVector{<:Index},
    k::Int;
    kwargs...,
)
    sites = isempty(instr.pt_sites) ? Index[pt_sites_arg...] : instr.pt_sites
    _validate_single_leg_sites("ObservableMeasurement", sites, instr.leg_plev)
    all(s -> _tstep_from_site(s) in (nothing, k), sites) || throw(
        ArgumentError("ObservableMeasurement: all pt_sites must have tstep=$k when tagged."),
    )
    phys_sites = Index[_phys_site_from_liouv(s) for s in sites]
    obs_h = MPO(instr.op, phys_sites) # build the observable in the Hilbert space
    obs_l = to_liouville(obs_h; sites=sites) # convert the observable to the Liouville space
    return _reindex_itensor(_mps_to_itensor(obs_l), siteinds(obs_l), sites)
end

function instrument_itensor(
    instr::TraceOut,
    pt_sites_arg::AbstractVector{<:Index},
    k::Int;
    kwargs...,
)
    sites = isempty(instr.pt_sites) ? Index[pt_sites_arg...] : instr.pt_sites
    _validate_single_leg_sites("TraceOut", sites, instr.leg_plev)
    all(s -> _tstep_from_site(s) in (nothing, k), sites) || throw(
        ArgumentError("TraceOut: all pt_sites must have tstep=$k when tagged."),
    )
    return _vectorized_identity_itensor(sites)
end

function instrument_itensor(
    instr::ProductInstrument,
    input_pt_sites::AbstractVector{<:Index},
    output_pt_sites::AbstractVector{<:Index},
    k::Int;
    kwargs...,
)
    T_out = instrument_itensor(instr.output_instr, output_pt_sites, k - 1; kwargs...)
    T_in = instrument_itensor(instr.input_instr, input_pt_sites, k; kwargs...)
    return T_in * T_out
end

function instrument_itensor(
    instr::IdentityOperation,
    input_pt_sites::AbstractVector{<:Index},
    output_pt_sites::AbstractVector{<:Index},
    k::Int;
    kwargs...,
)
    in_sites = isempty(instr.input_pt_sites) ? Index[input_pt_sites...] : instr.input_pt_sites
    out_sites = isempty(instr.output_pt_sites) ? Index[output_pt_sites...] : instr.output_pt_sites
    _validate_two_leg_map("IdentityOperation", in_sites, out_sites)
    all(s -> _tstep_from_site(s) in (nothing, k), in_sites) || throw(
        ArgumentError("IdentityOperation: all input_pt_sites must have tstep=$k when tagged."),
    )
    all(s -> _tstep_from_site(s) in (nothing, k - 1), out_sites) || throw(
        ArgumentError("IdentityOperation: all output_pt_sites must have tstep=$(k - 1) when tagged."),
    )
    map_t = ITensor(1.0)
    for (sin, sout) in zip(in_sites, out_sites)
        map_t *= delta(sin, sout)
    end
    return map_t
end

function instrument_itensor(
    instr::OpenOutput,
    pt_sites_arg::AbstractVector{<:Index},
    k::Int;
    kwargs...,
)
    sites = isempty(instr.pt_sites) ? Index[pt_sites_arg...] : instr.pt_sites
    if !isempty(sites)
        _validate_single_leg_sites("OpenOutput", sites, _OUTPUT_PLEV)
        all(s -> _tstep_from_site(s) in (nothing, k), sites) || throw(
            ArgumentError("OpenOutput: all pt_sites must have tstep=$k when tagged."),
        )
    end
    return ITensor(1.0)
end

function instrument_itensor(
    instr::OpenInput,
    pt_sites_arg::AbstractVector{<:Index},
    k::Int;
    kwargs...,
)
    sites = isempty(instr.pt_sites) ? Index[pt_sites_arg...] : instr.pt_sites
    if !isempty(sites)
        _validate_single_leg_sites("OpenInput", sites, _INPUT_PLEV)
        all(s -> _tstep_from_site(s) in (nothing, k), sites) || throw(
            ArgumentError("OpenInput: all pt_sites must have tstep=$k when tagged."),
        )
    end
    return ITensor(1.0)
end

# Evolve-slot entry point: pick the PT leg that matches `leg_plev`.
function instrument_itensor(
    instr::SingleLegInstrument,
    input_pt_sites::AbstractVector{<:Index},
    output_pt_sites::AbstractVector{<:Index},
    step::Int;
    kwargs...,
)
    if instr.leg_plev == _OUTPUT_PLEV
        return instrument_itensor(instr, output_pt_sites, step - 1; kwargs...)
    else
        return instrument_itensor(instr, input_pt_sites, step; kwargs...)
    end
end

function instrument_itensor(
    instr::OpenInOut,
    input_pt_sites::AbstractVector{<:Index},
    output_pt_sites::AbstractVector{<:Index},
    k::Int;
    kwargs...,
)
    in_sites = isempty(instr.input_pt_sites) ? Index[input_pt_sites...] : instr.input_pt_sites
    out_sites = isempty(instr.output_pt_sites) ? Index[output_pt_sites...] : instr.output_pt_sites
    if !isempty(in_sites) || !isempty(out_sites)
        _validate_two_leg_map("OpenInOut", in_sites, out_sites)
        all(s -> _tstep_from_site(s) in (nothing, k), in_sites) || throw(
            ArgumentError("OpenInOut: all input_pt_sites must have tstep=$k when tagged."),
        )
        all(s -> _tstep_from_site(s) in (nothing, k - 1), out_sites) || throw(
            ArgumentError("OpenInOut: all output_pt_sites must have tstep=$(k - 1) when tagged."),
        )
    end
    return ITensor(1.0)
end

_unitary_hamiltonian(H::OpSum, k::Int, dt::Real) = H

function _unitary_hamiltonian(H_of_t::Function, k::Int, dt::Real)
    H = H_of_t((k - 0.5) * dt)
    H isa OpSum || throw(
        ArgumentError("UnitaryPropagation: time-dependent Hamiltonian function must return an OpSum; got $(typeof(H))."),
    )
    return H
end

function instrument_itensor(
    instr::UnitaryPropagation,
    input_pt_sites::AbstractVector{<:Index},
    output_pt_sites::AbstractVector{<:Index},
    k::Int;
    dt::Real,
    alg=Trotter{2}(),
    kwargs...,
)
    in_sites = isempty(instr.input_pt_sites) ? Index[input_pt_sites...] : instr.input_pt_sites
    out_sites = isempty(instr.output_pt_sites) ? Index[output_pt_sites...] : instr.output_pt_sites
    _validate_two_leg_map("UnitaryPropagation", in_sites, out_sites)
    all(s -> _tstep_from_site(s) in (nothing, k), in_sites) || throw(
        ArgumentError("UnitaryPropagation: all input_pt_sites must have tstep=$k when tagged."),
    )
    all(s -> _tstep_from_site(s) in (nothing, k - 1), out_sites) || throw(
        ArgumentError("UnitaryPropagation: all output_pt_sites must have tstep=$(k - 1) when tagged."),
    )
    H = _unitary_hamiltonian(instr.H, k, dt)
    liouv_os = liouvillian_opsum(H, OpSum[])
    if isempty(ITensors.terms(liouv_os))
        id_map = ITensor(1.0)
        for (sin, sout) in zip(in_sites, out_sites)
            id_map *= delta(sin, sout)
        end
        return id_map
    end
    # liouvillian_propagator builds on canonical system Liouville sites (with `Site`
    # tag). PT legs drop `Site` to stay within ITensors' four-tag limit once `tstep=` is added.
    gate_sites = Index[instr.sites...]
    length(gate_sites) == length(out_sites) || throw(
        ArgumentError(
            "UnitaryPropagation: expected $(length(out_sites)) system sites, got $(length(gate_sites)).",
        ),
    )
    U_t = liouvillian_propagator(
        H,
        gate_sites,
        dt;
        alg=alg,
        jump_ops=OpSum[],
    )
    # U_t has unprimed gate_sites (output) and prime.(gate_sites) (input).
    # Relabel to PT leg convention: output → out_sites, input → in_sites.
    U_pt = U_t
    for (g, pt_out) in zip(gate_sites, out_sites)
        U_pt = replaceind(U_pt, g, pt_out)
    end
    for (g, pt_in) in zip(gate_sites, in_sites)
        U_pt = replaceind(U_pt, prime(g), pt_in)
    end
    return U_pt
end

function instrument_itensor(
    instr::CustomTwoLegInstrument,
    input_pt_sites::AbstractVector{<:Index},
    output_pt_sites::AbstractVector{<:Index},
    k::Int;
    kwargs...,
)
    runtime_in = Index[input_pt_sites...]
    runtime_out = Index[output_pt_sites...]
    _validate_two_leg_map("CustomTwoLegInstrument", runtime_in, runtime_out)
    all(s -> _tstep_from_site(s) in (nothing, k), runtime_in) || throw(
        ArgumentError("CustomTwoLegInstrument: all input_pt_sites must have tstep=$k when tagged."),
    )
    all(s -> _tstep_from_site(s) in (nothing, k - 1), runtime_out) || throw(
        ArgumentError("CustomTwoLegInstrument: all output_pt_sites must have tstep=$(k - 1) when tagged."),
    )

    if isempty(instr.source_input) && isempty(instr.source_output)
        return _reindex_itensor(
            _reindex_itensor(copy(instr.data), instr.input_pt_sites, runtime_in),
            instr.output_pt_sites,
            runtime_out,
        )
    end

    target_in = isempty(instr.input_pt_sites) ? runtime_in : instr.input_pt_sites
    target_out = isempty(instr.output_pt_sites) ? runtime_out : instr.output_pt_sites
    _validate_two_leg_map("CustomTwoLegInstrument", target_in, target_out)

    t = _reindex_itensor(
        _reindex_itensor(copy(instr.data), instr.source_input, target_in),
        instr.source_output,
        target_out,
    )
    return _reindex_itensor(_reindex_itensor(t, target_in, runtime_in), target_out, runtime_out)
end
function instrument_itensor(
    instr::AbstractInstrument,
    args...;
    kwargs...,
)
    throw(MethodError(instrument_itensor, (instr, args...)))
end

"""
    create_instruments(pt, seq; default, alg)

Materialize an `InstrumentSeq` as one `ITensor` per schedule slot.

The returned vector has length `pt.nsteps + 1`:
- index `1` is the `tstep = 0` preparation,
- indices `2:pt.nsteps` are evolve-slot instruments for steps `1:pt.nsteps-1`,
- index `pt.nsteps + 1` is the terminal instrument at `tstep = pt.nsteps`
  (`TraceOut`, `ObservableMeasurement`, or `OpenOutput` / open no-op).

Schedule rewriting (same pairing policy as `instrument_leg_maps`):
consecutive physical single-leg output/input entries become a
[`ProductInstrument`](@ref) at the output slot; the consumed input slot is
replaced by `default` when it still lies in the evolve range. A terminal
[`IdentityOperation`](@ref) becomes [`OpenOutput`](@ref).

Every slot is then materialized only through [`instrument_itensor`](@ref).
"""
function create_instruments(
    pt::ProcessTensors.ProcessTensor,
    seq::InstrumentSeq;
    default::AbstractInstrument=ProcessTensors._schedule_default_instr(pt),
    alg=Trotter{2}(),
)
    n = pt.nsteps
    slots = Vector{AbstractInstrument}(undef, n + 1)
    for t in 0:n
        slots[t + 1] = resolve_instrument(seq, t, default)
    end
    if slots[end] isa IdentityOperation
        slots[end] = open_output()
    end

    # Pair physical single-leg out/in (OpenOutput does not force a partner).
    for s in 1:(n - 1)
        out_entry = slots[s + 1]
        in_entry = slots[s + 2]
        out_entry isa SingleLegInstrument || continue
        out_entry.leg_plev == _OUTPUT_PLEV || continue
        out_entry isa OpenOutput && continue

        (in_entry isa SingleLegInstrument && in_entry.leg_plev == _INPUT_PLEV) || throw(
            ArgumentError(
                "create_instruments: single-leg output instrument $(typeof(out_entry)) at tstep=$s " *
                "requires a single-leg input instrument at tstep=$(s + 1); got $(typeof(in_entry)).",
            ),
        )
        slots[s + 1] = ProductInstrument(in_entry, out_entry)
        if s + 1 <= n - 1
            slots[s + 2] = default
        end
    end

    for s in 1:(n - 1)
        instr = slots[s + 1]
        if instr isa SingleLegInstrument && instr.leg_plev == _INPUT_PLEV && !(instr isa OpenInput)
            throw(
                ArgumentError(
                    "create_instruments: unpaired single-leg input instrument $(typeof(instr)) at tstep=$s; " *
                    "expected a single-leg output instrument at tstep=$(s - 1).",
                ),
            )
        end
    end

    terminal = slots[end]
    if terminal isa OpenInput || terminal isa OpenInOut || terminal isa TwoLegInstrument
        throw(
            ArgumentError(
                "create_instruments: $(typeof(terminal)) is not supported at terminal tstep=$n.",
            ),
        )
    elseif !(terminal isa SingleLegInstrument) || terminal.leg_plev != _OUTPUT_PLEV
        throw(
            ArgumentError(
                "create_instruments: unsupported terminal instrument $(typeof(terminal)) at tstep=$n.",
            ),
        )
    end

    ProcessTensors._validate_instrument_schedule!(pt, seq, default, "create_instruments")

    instruments = Vector{ITensor}(undef, n + 1)
    instruments[1] = instrument_itensor(slots[1], ProcessTensors.input_sites(pt, 0), 0)
    for step in 1:(n - 1)
        out_prev, in_curr = ProcessTensors.coupling_times(pt, step)
        instruments[step + 1] = instrument_itensor(
            slots[step + 1], in_curr, out_prev, step; dt=pt.dt, alg=alg,
        )
    end
    out_final, _ = ProcessTensors.coupling_times(pt, n)
    instruments[n + 1] = instrument_itensor(slots[end], out_final, n - 1)

    return instruments
end
