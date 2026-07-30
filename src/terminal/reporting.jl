# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: src/terminal/reporting.jl
# Contributor: Gauthameshwar S.
#
# Implements the run-reporter lifecycle: construction, spinner worker and
# thread-pool selection, stage/bar coordination, verbose checkpoints, and
# idempotent cleanup.

"""
    _run_reporter(progress, verbose; output=stderr) -> _AbstractRunReporter

Construct the reporter for one tracked operation.

- `progress=false` returns a strict no-op reporter (`verbose` logging only).
- `progress=true` returns an active TTY reporter on `output`.
- `progress=:auto` activates only for a suitable interactive terminal outside CI.

Construction never starts the spinner; `_progress_start!` owns startup.
"""
function _run_reporter(progress::Union{Bool,Symbol}, verbose::Bool; output::IO=stderr)
    enabled = if progress === :auto
        output isa Base.TTY && !haskey(ENV, "CI")
    elseif progress isa Bool
        progress
    else
        throw(ArgumentError("progress must be :auto, true, or false; got $progress."))
    end
    return enabled ? _TTYRunReporter(output, verbose) : _NoRunReporter(verbose)
end

# Splatting `pairs(meta)` keeps `@info` metadata dynamic while preserving the
# standard structured-log rendering.
function _verbose_info(message::AbstractString; meta...)
    @info message meta...
    return nothing
end

# --- No-op reporter: progress operations vanish, verbose checkpoints remain ---

function _progress_start!(run::_NoRunReporter, operation::AbstractString; meta...)
    run.verbose && _verbose_info(operation; meta...)
    return nothing
end

function _progress_stage!(run::_NoRunReporter, description::AbstractString; meta...)
    run.verbose && _verbose_info(description; meta...)
    return nothing
end

# A bar is itself a stage checkpoint; keep the durable log when verbose is on.
function _begin_progress_bar!(run::_NoRunReporter, description::AbstractString, total::Integer)
    run.verbose && _verbose_info(description; total=Int(total))
    return nothing
end

_progress_update!(::_NoRunReporter, ::Integer; values...) = nothing
_finish_progress_bar!(::_NoRunReporter) = nothing
_progress_finish!(::_NoRunReporter) = nothing

# --- Active TTY reporter ---

# Advance the glyph at most once per spinner interval, regardless of which
# writer (worker task or bar update) triggered the repaint.
function _advance_frame_if_due!(run::_TTYRunReporter, now::Float64)
    if now - run.last_frame_time >= _SPINNER_INTERVAL
        run.frame += 1
        run.last_frame_time = now
    end
    return nothing
end

function _paint_header!(run::_TTYRunReporter)
    _paint_spinner_line!(run.output, run.operation, run.description, run.frame, run.started)
    return nothing
end

# Wait between spinner frames.
#
# `Base.sleep` is libuv-based and is serviced on the main thread, so it does not
# wake while that thread is inside a BLAS foreign call. Background workers
# therefore use `Libc.systemsleep`. The single-thread cooperative path keeps
# `Base.sleep` so the spinner can still yield to scientific work.
function _spinner_wait(; use_systemsleep::Bool)
    if use_systemsleep
        Libc.systemsleep(_SPINNER_INTERVAL)
    else
        sleep(_SPINNER_INTERVAL)
    end
    return nothing
end

function _spinner_loop!(run::_TTYRunReporter; use_systemsleep::Bool)
    while run.running[]
        lock(run.io_lock) do
            run.running[] || return nothing
            run.state === _ProgressSpinner || run.state === _ProgressSpinnerBar || return nothing
            _advance_frame_if_due!(run, time())
            _paint_header!(run)
            return nothing
        end
        _spinner_wait(; use_systemsleep)
    end
    return nothing
end

# Pin a sticky task onto a concrete Julia thread so the scheduler cannot migrate
# it onto the main thread, where BLAS foreign calls would freeze it.
function _spawn_sticky_on_thread!(f, tid::Integer)
    task = Task(f)
    task.sticky = true
    ccall(:jl_set_task_tid, Cvoid, (Any, Cint), task, convert(Cint, tid - 1))
    schedule(task)
    return task
end

# Thread-pool policy for the spinner worker.
#
# When Julia is started with interactive threads (`-t M,N`), the main thread is
# moved into the interactive pool, so `:interactive` would park the spinner on
# the BLAS thread. Prefer `:default` in that case. With a plain `-t M` (M>1)
# pool, pin sticky to thread 2. Otherwise fall back to a cooperative task.
function _spawn_spinner_worker!(run::_TTYRunReporter)
    n_default = Threads.nthreads(:default)
    n_interactive = Threads.nthreads(:interactive)
    if n_default + n_interactive <= 1
        run.task = @async _spinner_loop!(run; use_systemsleep=false)
    elseif n_interactive > 0
        run.task = Threads.@spawn :default _spinner_loop!(run; use_systemsleep=true)
    else
        run.task = _spawn_sticky_on_thread!(
            () -> _spinner_loop!(run; use_systemsleep=true),
            2,
        )
    end
    return nothing
end

# Erase the transient block (bar line first, then header) so a persistent log
# can be emitted from a clean line. Cursor ends at column 0 of the header line.
function _clear_transient!(run::_TTYRunReporter)
    run.bar isa _ProgressMeterBar && _erase_bar!(run.output, run.bar)
    _erase_current_line!(run.output)
    return nothing
end

function _repaint_transient!(run::_TTYRunReporter)
    _paint_header!(run)
    run.bar isa _ProgressMeterBar && _repaint_bar!(run.bar)
    return nothing
end

# Emit one persistent checkpoint without corrupting the live display: clear,
# log, redraw — all under the terminal lock so the worker cannot interleave.
function _checkpoint_with_display!(run::_TTYRunReporter, message::AbstractString; meta...)
    _clear_transient!(run)
    _verbose_info(message; meta...)
    _repaint_transient!(run)
    return nothing
end

function _progress_start!(run::_TTYRunReporter, operation::AbstractString; meta...)
    run.state === _ProgressIdle || return nothing
    run.verbose && _verbose_info(operation; meta...)
    lock(run.io_lock) do
        run.operation = String(operation)
        run.description = ""
        run.started = time()
        run.frame = 0
        run.last_frame_time = run.started
        run.cursor_hidden = _hide_cursor!(run.output)
        run.state = _ProgressSpinner
        _paint_header!(run)
        return nothing
    end
    run.running[] = true
    _spawn_spinner_worker!(run)
    return nothing
end

function _progress_stage!(run::_TTYRunReporter, description::AbstractString; meta...)
    if run.state === _ProgressIdle || run.state === _ProgressClosed
        run.verbose && _verbose_info(description; meta...)
        return nothing
    end
    lock(run.io_lock) do
        run.description = String(description)
        if run.verbose
            _checkpoint_with_display!(run, description; meta...)
        else
            _paint_header!(run)
        end
        return nothing
    end
    return nothing
end

function _begin_progress_bar!(run::_TTYRunReporter, description::AbstractString, total::Integer)
    # A bar is itself a stage: update the header and emit the verbose checkpoint.
    _progress_stage!(run, description; total=Int(total))
    run.state === _ProgressSpinner || return nothing
    lock(run.io_lock) do
        run.bar = _create_bar(run.output, description, total)
        run.state = _ProgressSpinnerBar
        return nothing
    end
    return nothing
end

function _progress_update!(run::_TTYRunReporter, position::Integer; values...)
    run.state === _ProgressSpinnerBar || return nothing
    lock(run.io_lock) do
        bar = run.bar
        bar isa _ProgressMeterBar || return nothing
        _update_bar!(bar, position; values=values)
        # Bar repaints return the cursor to the header line; refresh the spinner
        # there so single-thread runs stay animated between worker wakeups.
        _advance_frame_if_due!(run, time())
        _paint_header!(run)
        return nothing
    end
    return nothing
end

function _finish_progress_bar!(run::_TTYRunReporter)
    run.state === _ProgressSpinnerBar || return nothing
    lock(run.io_lock) do
        bar = run.bar
        if bar isa _ProgressMeterBar
            _erase_bar!(run.output, bar)
        end
        run.bar = _NoBar()
        run.state = _ProgressSpinner
        _paint_header!(run)
        return nothing
    end
    return nothing
end

function _progress_finish!(run::_TTYRunReporter)
    run.state === _ProgressClosed && return nothing
    if run.state === _ProgressIdle
        run.state = _ProgressClosed
        return nothing
    end
    # Stop the worker before touching the display; waiting happens outside the
    # lock because the worker acquires it for every frame.
    run.running[] = false
    task = run.task
    if task !== nothing && !istaskdone(task)
        try
            wait(task)
        catch
        end
    end
    run.task = nothing
    lock(run.io_lock) do
        run.bar isa _ProgressMeterBar && _erase_bar!(run.output, run.bar)
        run.bar = _NoBar()
        _erase_current_line!(run.output)
        run.cursor_hidden && _show_cursor!(run.output)
        run.cursor_hidden = false
        run.state = _ProgressClosed
        flush(run.output)
        return nothing
    end
    return nothing
end
