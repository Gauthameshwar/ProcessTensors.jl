# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: src/process_tensor/multitime.jl
# Contributor: Gauthameshwar S.
#
# Builds instrument schedules for sequential multi-time process-tensor
# correlation functions.

"""
    two_time_correlation_seq(pt, (O_A, j_A), (O_B, j_B); rho0, default_instr)

Build an `InstrumentSeq` for the two-time correlator
``\\langle A(t_A) B(t_B)\\rangle`` with ``t = (j - 1) dt``.

`j_A` and `j_B` are snapshot indices in `1:pt.nsteps+1`, matching the samples of
[`evolve`](@ref): snapshot `1` is the initial state at time `0` and composes
onto the `rho0` preparation, snapshot `pt.nsteps + 1` is the output of the final
core, and any other snapshot `j` is the between-attachment `j - 1` after core
`j - 1`.

# Examples
```julia
seq = two_time_correlation_seq(pt, (O_A, 4), (O_B, 1); rho0=ρ0)  # ⟨A(3dt) B(0)⟩
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
    O_A, j_A = op_a
    O_B, j_B = op_b
    n = pt.nsteps
    for j in (j_A, j_B)
        1 <= j <= n + 1 || throw(
            ArgumentError(
                "two_time_correlation_seq: snapshot index must lie in 1:$(n + 1); got $j.",
            ),
        )
    end

    phys_sites = _phys_sites_from_hilbert_state(rho0)
    j_early = min(j_A, j_B)
    j_late = max(j_A, j_B)
    terminal_late = j_late == n + 1

    seq = InstrumentSeq(default=default_instr, nsteps=n)

    if j_A == j_B
        if j_late == 1
            same_time = observable_measurement(O_B; leg_plev=1) *
                        observable_measurement(O_A; leg_plev=1) *
                        state_preparation(rho0)
            add!(seq, same_time, 1)
            add!(seq, trace_out(), n)
            return seq
        end
        add!(seq, state_preparation(rho0), 1)
        # Terminal single-leg and interior left_action compose observables in opposite order;
        # B*A factors → Tr(A B ρ) via left_action, A*B factors → Tr(A B ρ) on the terminal leg.
        same_time = terminal_late ?
                    observable_measurement(O_A) * observable_measurement(O_B) :
                    observable_measurement(O_B) * observable_measurement(O_A)
        if terminal_late
            add!(seq, same_time, n)
        else
            add!(seq, left_action(same_time, phys_sites), j_late - 1)
            for k in j_late:(n - 1)
                add!(seq, identity_operation(), k)
            end
            add!(seq, trace_out(), n)
        end
    else
        if j_A > j_B
            O_early, O_late, early_side = O_B, O_A, :left
        else
            O_early, O_late, early_side = O_A, O_B, :right
        end

        if j_early == 1
            prep = early_side === :left ?
                   observable_measurement(O_early; leg_plev=1) * state_preparation(rho0) :
                   state_preparation(rho0) * observable_measurement(O_early; leg_plev=1)
            add!(seq, prep, 1)
        else
            add!(seq, state_preparation(rho0), 1)
            early_lr = early_side === :left ? left_action(O_early, phys_sites) : right_action(O_early, phys_sites)
            add!(seq, early_lr, j_early - 1)
        end

        if terminal_late
            add!(seq, observable_measurement(O_late), n)
        else
            add!(seq, left_action(O_late, phys_sites), j_late - 1)
            for k in j_late:(n - 1)
                add!(seq, identity_operation(), k)
            end
            add!(seq, trace_out(), n)
        end
    end

    return seq
end
