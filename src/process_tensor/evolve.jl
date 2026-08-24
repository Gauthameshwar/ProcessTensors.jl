# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: src/process_tensor/evolve.jl
# Contributor: Gauthameshwar S.
#
# Implements reduced-system evolution by contracting process tensors with
# instrument schedules.

import ITensors.Ops: Trotter

# Maximally mixed system state |I/d⟩⟩ on a single Liouville process-tensor site.
# Used to CPTP-close future input legs when taking an intermediate marginal on a
# full-length process tensor (see `_evolve_snapshot_seq`).
function _maximally_mixed_liouville(liouv_site::Index)
    d2 = dim(liouv_site)
    d = isqrt(d2)
    d * d == d2 || throw(
        ArgumentError(
            "_maximally_mixed_liouville: expected Liouville dim d^2; got dim=$d2.",
        ),
    )
    mixed = Instruments._vectorized_identity_itensor(Index[liouv_site]) / d
    return _liouville_mps_from_itensor(mixed, Index[liouv_site])
end

# Build the InstrumentSeq whose `evaluate_process` result is the reduced system
# state at snapshot `k` (`t = k * dt`, open output leg `out_k`).
#
# Past slots `0:k` are copied from `seq`. The output at time `k` is left open.
# Every *future* core is closed with the disconnected CPTP probe
# `TraceOut ⊗ StatePreparation(I/d)` so dangling compressed memory bonds
# (ACE) are contracted through the remaining influence functional. This is the
# same object as the O(N) `:closures` walk, evaluated once per snapshot
# (O(N²) total).
function _evolve_snapshot_seq(
    pt::ProcessTensor,
    seq::InstrumentSeq,
    k::Int;
    default_instr::AbstractInstrument=_schedule_default_instr(pt),
)
    n = pt.nsteps
    0 <= k < n || throw(BoundsError(0:(n - 1), k))

    snap = InstrumentSeq(default=default_instr, nsteps=n)
    for t in 0:k
        instr = if t == 0
            Instruments.resolve_instrument(seq, 0)
        else
            Instruments.resolve_instrument(seq, t, default_instr)
        end
        instr === nothing && throw(
            ArgumentError(
                "evolve: missing instrument at tstep=$t while building snapshot k=$k.",
            ),
        )
        add!(snap, instr, t)
    end

    if k == n - 1
        add!(snap, open_output(), n)
        return snap
    end

    ρ_mm = _maximally_mixed_liouville(pt.coupling_site)
    # Slot k+1 claims out_k (left open) and closes in_{k+1} with I/d.
    add!(snap, open_output() * state_preparation(ρ_mm), k + 1)
    for t in (k + 2):(n - 1)
        add!(snap, trace_out() * state_preparation(ρ_mm), t)
    end
    add!(snap, trace_out(), n)
    return snap
end

# Right environments for intermediate evolve snapshots.
# After SVD compression the memory bond is not a bath Liouville basis, so it
# cannot be closed with a bare vec(I). The correct vector is the CPTP trace
# through all later cores: c_j = (core_{j+1} · c_{j+1} · ⟨⟨I|_out · ⟨⟨I|_in) / d.
function _pt_bond_closures(pt::ProcessTensor)
    n = pt.nsteps
    closures = Vector{ITensor}(undef, max(n - 1, 0))
    n > 1 || return closures
    carry = ITensor(1.0)
    for j in n:-1:2
        out_leg = only(output_sites(pt, j - 1))
        in_leg = prime(out_leg)
        d_sys = isqrt(dim(out_leg))
        blob = pt.core[j] * carry
        blob *= Instruments._vectorized_identity_itensor(Index[out_leg])
        if hasind(blob, in_leg)
            blob *= Instruments._vectorized_identity_itensor(Index[in_leg])
        end
        blob /= d_sys
        carry = blob
        closures[j - 1] = blob
    end
    return closures
end

function _evolve(
    ::Val{:evaluate},
    pt::ProcessTensor,
    seq::InstrumentSeq;
    default_instr::AbstractInstrument,
    alg,
    run,
)
    states_liouville = Vector{MPS{Liouville}}(undef, pt.nsteps)
    states_hilbert = Vector{MPO{Hilbert}}(undef, pt.nsteps)
    times = [pt.dt * k for k in 0:(pt.nsteps - 1)]

    @progress_stage run "Preparing trajectory"
    @progress_bar run "Computing reduced snapshots" pt.nsteps begin
        for k in 0:(pt.nsteps - 1)
            snap = _evolve_snapshot_seq(pt, seq, k; default_instr=default_instr)
            rho_liouv = _evaluate_process(
                pt,
                snap;
                default_instr=default_instr,
                alg=alg,
                run=_NO_RUN_REPORTER,
            )
            rho_liouv isa MPS{Liouville} || throw(
                ArgumentError(
                    "evolve: expected MPS{Liouville} at snapshot k=$k; got $(typeof(rho_liouv)).",
                ),
            )
            states_liouville[k + 1] = rho_liouv
            states_hilbert[k + 1] = to_hilbert(rho_liouv)
            snapshot = k + 1
            @progress_update run snapshot (t=times[snapshot],)
        end
    end
    return (times=times, states_liouville=states_liouville, states_hilbert=states_hilbert)
end

function _evolve(
    ::Val{:closures},
    pt::ProcessTensor,
    seq::InstrumentSeq;
    default_instr::AbstractInstrument,
    alg,
    run,
)
    states_liouville = Vector{MPS{Liouville}}(undef, pt.nsteps)
    states_hilbert = Vector{MPO{Hilbert}}(undef, pt.nsteps)
    times = [pt.dt * k for k in 0:(pt.nsteps - 1)]

    @progress_stage run "Preparing trajectory"
    instruments = Instruments._create_instruments(
        pt,
        seq;
        default=default_instr,
        alg=alg,
        run=_NO_RUN_REPORTER,
    )

    @progress_stage run "Computing bond closures"
    closures = _pt_bond_closures(pt)

    prev_pt_core = pt.core[1] * instruments[1]
    @progress_bar run "Computing reduced snapshots" pt.nsteps begin
        for k in 0:(pt.nsteps - 1)
            if k > 0
                prev_pt_core *= instruments[k + 1]
                prev_pt_core *= pt.core[k + 1]
            end
            out_sites = output_sites(pt, k)
            snapshot = prev_pt_core
            if k + 1 <= length(closures)
                snapshot = snapshot * closures[k + 1]
            end
            # Close leftover legs (Dense fused-bath traces; ACE should already
            # be absorbed by the closures) except the open system output.
            reduced = _trace_out_except(snapshot, out_sites)
            rho_liouv = _liouville_mps_from_itensor(reduced, out_sites)
            states_liouville[k + 1] = rho_liouv
            states_hilbert[k + 1] = to_hilbert(rho_liouv)
            snapshot_k = k + 1
            @progress_update run snapshot_k (t=times[snapshot_k],)
        end
    end
    return (times=times, states_liouville=states_liouville, states_hilbert=states_hilbert)
end

# Multiple dispatch helpers for evolve storage depending on what trajectories the user wants
function _evolve_storage(::Val{false}, ::Val{false}, nsteps::Int)
    return (
        states_liouville=Vector{MPS{Liouville}}(undef, nsteps),
        states_hilbert=Vector{MPO{Hilbert}}(undef, nsteps),
    )
end

function _evolve_storage(::Val{true}, ::Val{false}, nsteps::Int)
    return (
        states_liouville=Vector{MPS{Liouville}}(undef, nsteps),
        states_hilbert=Vector{MPO{Hilbert}}(undef, nsteps),
        tester_states_liouville=Vector{MPS{Liouville}}(undef, nsteps),
        tester_states_hilbert=Vector{MPO{Hilbert}}(undef, nsteps),
    )
end

function _evolve_storage(::Val{false}, ::Val{true}, nsteps::Int)
    return (
        states_liouville=Vector{MPS{Liouville}}(undef, nsteps),
        states_hilbert=Vector{MPO{Hilbert}}(undef, nsteps),
        joint_states_liouville=Vector{MPS{Liouville}}(undef, nsteps),
        joint_states_hilbert=Vector{MPO{Hilbert}}(undef, nsteps),
    )
end

function _evolve_storage(::Val{true}, ::Val{true}, nsteps::Int)
    return (
        states_liouville=Vector{MPS{Liouville}}(undef, nsteps),
        states_hilbert=Vector{MPO{Hilbert}}(undef, nsteps),
        tester_states_liouville=Vector{MPS{Liouville}}(undef, nsteps),
        tester_states_hilbert=Vector{MPO{Hilbert}}(undef, nsteps),
        joint_states_liouville=Vector{MPS{Liouville}}(undef, nsteps),
        joint_states_hilbert=Vector{MPO{Hilbert}}(undef, nsteps),
    )
end

function _store_evolve_snapshot!(
    storage,
    snapshot::ITensor,
    system_sites::AbstractVector{<:Index},
    tester_sites::AbstractVector{<:Index},
    slot::Int,
    ::Val{ReturnTester},
    ::Val{ReturnJoint},
) where {ReturnTester,ReturnJoint}
    system_state = _liouville_mps_from_itensor(
        _trace_out_except(snapshot, system_sites),
        system_sites,
    )
    storage.states_liouville[slot] = system_state
    storage.states_hilbert[slot] = to_hilbert(system_state)

    if ReturnTester
        tester_state = _liouville_mps_from_itensor(
            _trace_out_except(snapshot, tester_sites),
            tester_sites,
        )
        storage.tester_states_liouville[slot] = tester_state
        storage.tester_states_hilbert[slot] = to_hilbert(tester_state)
    end

    if ReturnJoint
        joint_sites = vcat(system_sites, tester_sites)
        joint_state = _liouville_mps_from_itensor(
            _trace_out_except(snapshot, joint_sites),
            joint_sites,
        )
        storage.joint_states_liouville[slot] = joint_state
        storage.joint_states_hilbert[slot] = to_hilbert(joint_state)
    end
    return nothing
end

function _evolve(
    ::Val{:closures},
    pt::ProcessTensor,
    seq::InstrumentSeq,
    return_tester::Val,
    return_joint::Val;
    default_instr::AbstractInstrument,
    alg,
    tester::Tester,
    tester_seq::Union{Nothing,TesterSeq},
    run,
)
    times = [pt.dt * k for k in 0:(pt.nsteps - 1)]
    storage = _evolve_storage(return_tester, return_joint, pt.nsteps)

    @progress_stage run "Preparing trajectory"
    instruments = Instruments._create_instruments(
        pt,
        seq;
        default=default_instr,
        alg=alg,
        run=_NO_RUN_REPORTER,
    )
    control = Instruments._compile_tester_control_data(
        pt,
        instruments,
        tester,
        tester_seq,
        "evolve",
    )

    @progress_stage run "Computing bond closures"
    closures = _pt_bond_closures(pt)

    prefix = pt.core[1] * control.instruments[1]
    @progress_bar run "Computing reduced snapshots" pt.nsteps begin
        for k in 0:(pt.nsteps - 1)
            if k > 0
                prefix *= control.instruments[k + 1]
                prefix *= pt.core[k + 1]
            end

            system_sites = output_sites(pt, k)
            snapshot = prefix * control.pending_post[k + 1]
            post_system = control.post_system[k + 1]
            if post_system !== nothing
                snapshot = replaceind(snapshot, post_system, only(system_sites))
            end
            snapshot = replaceind(
                snapshot,
                control.memory_output[k + 1],
                only(control.tester_sites),
            )
            if k + 1 <= length(closures)
                snapshot *= closures[k + 1]
            end

            slot = k + 1
            _store_evolve_snapshot!(
                storage,
                snapshot,
                system_sites,
                control.tester_sites,
                slot,
                return_tester,
                return_joint,
            )
            @progress_update run slot (t=times[slot],)
        end
    end
    return (; times, storage...)
end

function _evolve(
    ::Val{:evaluate},
    ::ProcessTensor,
    ::InstrumentSeq,
    ::Val,
    ::Val;
    tester::Tester,
    kwargs...,
)
    throw(
        ArgumentError(
            "evolve: tester trajectories require contraction=:closures; " *
            "contraction=:evaluate traces tester memory at each schedule end.",
        ),
    )
end

function _evolve(
    ::Val{S},
    ::ProcessTensor,
    ::InstrumentSeq,
    ::Val,
    ::Val;
    kwargs...,
) where {S}
    throw(
        ArgumentError(
            "evolve: contraction must be :evaluate or :closures; got :$S.",
        ),
    )
end

function _evolve(::Val{S}, pt::ProcessTensor, seq::InstrumentSeq; kwargs...) where {S}
    throw(
        ArgumentError(
            "evolve: contraction must be :evaluate or :closures; got :$S.",
        ),
    )
end

"""
    evolve(pt, seq; tester=nothing, tester_seq=nothing,
           return_tester=nothing, return_joint=false,
           default_instr=_schedule_default_instr(pt), contraction=:closures,
           alg=Trotter{2}(),
           progress=:auto, verbose=false)

Return reduced system snapshots generated by contracting a process tensor with
an instrument schedule.

Pass `tester` and optionally `tester_seq` to propagate ancillary tester memory.
Tester states are returned by default when a tester is present; set
`return_tester=false` to omit them or `return_joint=true` to additionally return
joint system-tester states. 

Tester trajectories use the fields `tester_states_liouville` and
`tester_states_hilbert`; requested joint trajectories use
`joint_states_liouville` and `joint_states_hilbert`. The system fields
`states_liouville` and `states_hilbert` are always returned.

`contraction` selects the contraction algorithm:

- `:closures` (default): one backward environment sweep plus one forward prefix
  walk, ``O(N)`` in the number of timesteps.
- `:evaluate`: one [`evaluate_process`](@ref) call per snapshot. Past instruments
  are copied from `seq`, `out_k` is left open, and every future core is
  CPTP-closed with `TraceOut ⊗ StatePreparation(I/d)`. Cost is ``O(N^2)``.

# Examples
```julia
trajectory = evolve(pt, ρ0)
ρ_t = trajectory.states_hilbert[3]
trajectory_ref = evolve(pt, ρ0; contraction=:evaluate)

sA = siteinds("Qubit", 1)
memory = tester(sA, to_dm(MPS(sA, ["0"])))
tester_trajectory = evolve(pt, ρ0; tester=memory)
```
"""
function evolve(
    pt::ProcessTensor,
    seq::InstrumentSeq;
    default_instr::AbstractInstrument=_schedule_default_instr(pt),
    contraction::Symbol=:closures,
    alg=Trotter{2}(),
    tester::Union{Nothing,Tester}=nothing,
    tester_seq::Union{Nothing,TesterSeq}=nothing,
    return_tester::Union{Nothing,Bool}=nothing,
    return_joint::Bool=false,
    progress::Union{Bool,Symbol}=:auto,
    verbose::Bool=false,
)
    _validate_instrument_schedule!(pt, seq, default_instr, "evolve")
    tester === nothing && tester_seq !== nothing && throw(
        ArgumentError("evolve: tester_seq requires a tester."),
    )
    want_tester = something(return_tester, tester !== nothing)
    (want_tester || return_joint) && tester === nothing && throw(
        ArgumentError(
            "evolve: return_tester=true or return_joint=true requires a tester.",
        ),
    )
    started = time()
    run = @progress_start progress verbose "Evolving reduced system" (
        nsteps=pt.nsteps,
        dt=pt.dt,
        contraction=contraction,
    )
    try
        result = if tester === nothing
            _evolve(
                Val(contraction),
                pt,
                seq;
                default_instr=default_instr,
                alg=alg,
                run=run,
            )
        else
            _evolve(
                Val(contraction),
                pt,
                seq,
                Val(want_tester),
                Val(return_joint);
                default_instr=default_instr,
                alg=alg,
                tester=tester,
                tester_seq=tester_seq,
                run=run,
            )
        end
        @progress_stage run "Evolved reduced system" (
            nsteps=pt.nsteps,
            dt=pt.dt,
            snapshots=pt.nsteps,
            elapsed_seconds=(time() - started),
        )
        return result
    finally
        @progress_finish run
    end
end

"""
    evolve(pt, rho0, seq; default_instr=_schedule_default_instr(pt))

Insert `state_preparation(rho0)` at `tstep = 0` and return reduced system
snapshots for the resulting schedule.
"""
function evolve(
    pt::ProcessTensor,
    rho0,
    seq::InstrumentSeq;
    default_instr::AbstractInstrument=_schedule_default_instr(pt),
    kwargs...
)
    seq_full = InstrumentSeq(seq.default, seq.nsteps; entries=Dict{Int,AbstractInstrument}(pairs(seq.entries)))
    add!(seq_full, state_preparation(rho0), 0)
    return evolve(pt, seq_full; default_instr=default_instr, kwargs...)
end

"""
    evolve(pt, rho0; default_instr=_schedule_default_instr(pt))

Return reduced system snapshots from an initial state using the default
instrument schedule.
"""
function evolve(
    pt::ProcessTensor,
    rho0;
    default_instr::AbstractInstrument=_schedule_default_instr(pt),
    kwargs...
)
    seq = InstrumentSeq(default=default_instr, nsteps=pt.nsteps)
    add!(seq, state_preparation(rho0), 0)
    return evolve(pt, seq; default_instr=default_instr, kwargs...)
end
