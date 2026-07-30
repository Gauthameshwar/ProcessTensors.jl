# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: src/terminal/types.jl
# Contributor: Gauthameshwar S.
#
# Defines run-reporter and bar-state types, plus the explicit display state
# machine, for the ProcessTensors.jl terminal progress backend.

"""
    _AbstractRunReporter

Internal supertype for one tracked-operation reporter.

A reporter is created once per public workflow through [`@progress_start`](@ref)
and threaded to internal implementations through a `run` keyword. Every
lifecycle helper (`_progress_start!`, `_progress_stage!`, `_begin_progress_bar!`,
`_progress_update!`, `_finish_progress_bar!`, `_progress_finish!`) dispatches on
this supertype.
"""
abstract type _AbstractRunReporter end

"""
    _NoRunReporter(verbose)

Reporter for progress-disabled runs.

All transient-display operations are strict no-ops: no task, no lock, no
progress meter, no terminal write, and no ANSI code is ever produced. Persistent
verbose checkpoints remain available when `verbose=true`.
"""
struct _NoRunReporter <: _AbstractRunReporter
    verbose::Bool
end

"""
    _NO_RUN_REPORTER

Shared silent reporter (`progress=false, verbose=false`) used as the default
`run` value for internal workers called outside a tracked public workflow.
"""
const _NO_RUN_REPORTER = _NoRunReporter(false)

@enum _ProgressDisplayState begin
    _ProgressIdle
    _ProgressSpinner
    _ProgressSpinnerBar
    _ProgressClosed
end

"""
    _AbstractBarState

Internal supertype for the deterministic-bar slot of a TTY reporter. Exactly one
bar may be active at a time; `_NoBar` marks the spinner-only display.
"""
abstract type _AbstractBarState end

struct _NoBar <: _AbstractBarState end

"""
    _TTYRunReporter

Active reporter that owns the two-line transient display:

```text
⠹ <operation> — <description>    Time: 0:00:08
<bar description>  43%|███████            |  ETA: 0:00:05  t=2.15  χ=128
```

State transitions follow `_ProgressDisplayState`:
`Idle → Spinner ↔ SpinnerBar → Closed`. The spinner worker task and every
terminal write share `io_lock`; `running` is the atomic stop flag for the
worker.
"""
mutable struct _TTYRunReporter <: _AbstractRunReporter
    output::IO
    state::_ProgressDisplayState
    operation::String
    description::String
    started::Float64
    frame::Int
    last_frame_time::Float64
    running::Threads.Atomic{Bool}
    task::Union{Nothing,Task}
    io_lock::ReentrantLock
    bar::_AbstractBarState
    cursor_hidden::Bool
    verbose::Bool
end

function _TTYRunReporter(output::IO, verbose::Bool)
    return _TTYRunReporter(
        output,
        _ProgressIdle,
        "",
        "",
        0.0,
        0,
        0.0,
        Threads.Atomic{Bool}(false),
        nothing,
        ReentrantLock(),
        _NoBar(),
        false,
        verbose,
    )
end
