# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: src/process_tensor/multitime.jl
# Contributor: Gauthameshwar S.
#
# Builds instrument schedules for sequential multi-time process-tensor
# correlation functions.

"""
    two_time_correlation_seq(pt, (O_A, n_A), (O_B, n_B); rho0, default_instr)

Build an `InstrumentSeq` for the two-time correlator
``\\langle A(t_A) B(t_B)\\rangle``.

`rho0` is prepared at `tstep = 0`, and operator insertions are placed on the
process-tensor legs indexed by `n_A` and `n_B`.

# Examples
```julia
seq = two_time_correlation_seq(pt, (O_A, 1), (O_B, 3); rho0=ρ0)
result = evaluate_process(pt, seq)
```
"""
function two_time_correlation_seq(
    pt::ProcessTensor,
    op_a::Tuple{OpSum, Int},
    op_b::Tuple{OpSum, Int};
    rho0::Union{AbstractMPO{Hilbert}, AbstractMPS{Hilbert}},
    default_instr::AbstractInstrument=_schedule_default_instr(pt),
)
    O_A, n_A = op_a
    O_B, n_B = op_b
    n_A >= 0 || throw(ArgumentError("two_time_correlation_seq: time index n_A must be ≥ 0; got $n_A."))
    n_B >= 0 || throw(ArgumentError("two_time_correlation_seq: time index n_B must be ≥ 0; got $n_B."))
    n_late = max(n_A, n_B)
    n_late + 1 <= pt.nsteps || throw(
        ArgumentError(
            "two_time_correlation_seq: max(n_A, n_B) + 1 = $(n_late + 1) exceeds pt.nsteps=$(pt.nsteps).",
        ),
    )

    phys_sites = _phys_sites_from_hilbert_state(rho0)
    n_early = min(n_A, n_B)
    slot_late = n_late + 1
    terminal_late = slot_late == pt.nsteps

    seq = InstrumentSeq(default=default_instr, nsteps=pt.nsteps)

    if n_A == n_B
        add!(seq, StatePreparation(rho0), 0)
        # Terminal single-leg and interior left_action compose observables in opposite order;
        # B*A factors → Tr(A B ρ) via left_action, A*B factors → Tr(A B ρ) on the terminal leg.
        same_time = terminal_late ?
                    ObservableMeasurement(O_A) * ObservableMeasurement(O_B) :
                    ObservableMeasurement(O_B) * ObservableMeasurement(O_A)
        if terminal_late
            add!(seq, same_time, pt.nsteps)
        else
            add!(seq, left_action(same_time, phys_sites), slot_late)
            for step in (slot_late + 1):(pt.nsteps - 1)
                add!(seq, IdentityOperation(), step)
            end
            add!(seq, TraceOut(), pt.nsteps)
        end
    else
        if n_A > n_B
            O_early, O_late, early_side = O_B, O_A, :left
        else
            O_early, O_late, early_side = O_A, O_B, :right
        end

        if n_early == 0
            prep = early_side === :left ?
                   ObservableMeasurement(O_early; leg_plev=1) * StatePreparation(rho0) :
                   StatePreparation(rho0) * ObservableMeasurement(O_early; leg_plev=1)
            add!(seq, prep, 0)
        else
            add!(seq, StatePreparation(rho0), 0)
            early_slot = n_early + 1
            early_lr = early_side === :left ? left_action(O_early, phys_sites) : right_action(O_early, phys_sites)
            add!(seq, early_lr, early_slot)
        end

        if terminal_late
            add!(seq, ObservableMeasurement(O_late), pt.nsteps)
        else
            add!(seq, left_action(O_late, phys_sites), slot_late)
            for step in (slot_late + 1):(pt.nsteps - 1)
                add!(seq, IdentityOperation(), step)
            end
            add!(seq, TraceOut(), pt.nsteps)
        end
    end

    return seq
end
