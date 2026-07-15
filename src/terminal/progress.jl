# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: src/terminal/progress.jl
# Contributor: Gauthameshwar S.
#
# Provides private ProgressMeter reporting scopes and terminal-safe cleanup for
# long-running ProcessTensors.jl workflows.

import ProgressMeter
import LinearAlgebra

const _SPINNER_GLYPHS = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏") #('◐', '◓', '◑', '◒')
const _SPINNER_COLOR = :green
const _BAR_COLOR = :cyan

mutable struct _ProcessSpinner
    dir::String
    run_flag::String
    desc_file::String
    pause_file::String
    proc::Base.Process
end

mutable struct _ProgressReporter
    enabled::Bool
    output::IO
    spinner_active::Bool
    spinner_desc::String
    spinner_t0::Float64
    spinner_frame::Int
    bar::Union{Nothing,ProgressMeter.Progress}
    spinner_running::Union{Nothing,Base.RefValue{Bool}}
    spinner_task::Union{Nothing,Task}
    process_spinner::Union{Nothing,_ProcessSpinner}
    io_lock::ReentrantLock
    blas_threads_saved::Union{Nothing,Int}
end

function _progress_reporter(progress::Union{Bool,Symbol}=:auto; output::IO=stderr)
    enabled = if progress === :auto
        output isa Base.TTY && !haskey(ENV, "CI")
    elseif progress isa Bool
        progress
    else
        throw(ArgumentError("progress must be :auto, true, or false; got $progress."))
    end
    return _ProgressReporter(
        enabled, output, false, "", 0.0, 0, nothing, nothing, nothing, nothing,
        ReentrantLock(), nothing,
    )
end

_child_reporter(reporter::_ProgressReporter) =
    _ProgressReporter(
        false, reporter.output, false, "", 0.0, 0, nothing, nothing, nothing, nothing,
        ReentrantLock(), nothing,
    )

_resolve_showvalues(showvalues) = showvalues isa Function ? showvalues() : showvalues

function _throttle_blas_for_spinner!(reporter::_ProgressReporter)
    reporter.blas_threads_saved !== nothing && return nothing
    saved = LinearAlgebra.BLAS.get_num_threads()
    reserved = max(1, saved - 1)
    reserved == saved && return nothing
    LinearAlgebra.BLAS.set_num_threads(reserved)
    reporter.blas_threads_saved = saved
    return nothing
end

function _restore_blas_threads!(reporter::_ProgressReporter)
    saved = reporter.blas_threads_saved
    saved === nothing && return nothing
    LinearAlgebra.BLAS.set_num_threads(saved)
    reporter.blas_threads_saved = nothing
    return nothing
end

function _finish_bar!(meter::ProgressMeter.Progress)
    if meter.counter >= meter.n
        meter.counter = max(meter.n - 1, 0)
    end
    # ProgressMeter leaves `numprintedvalues` unchanged when `showvalues=()`. A
    # keep=false finish then moves the cursor up by `offset + numprintedvalues`
    # even though no value lines were reprinted, overshooting the header line.
    # Zero the count first so finish returns the cursor to the header/bar line.
    meter.numprintedvalues = 0
    ProgressMeter.update!(meter, meter.n; keep=false, force=true, color=_BAR_COLOR, showvalues=())
    return nothing
end

function _erase_bar_line!(io::IO)
    print(io, "\e[B\r\e[2K\e[A")
    flush(io)
    return nothing
end

# Erase the ProgressMeter block below the header (bar + optional showvalues lines).
# Caller must leave the cursor on the header line when `offset > 0`, or on the bar
# line when `offset == 0`.
function _erase_progress_block!(io::IO, offset::Integer, numprintedvalues::Integer)
    npv = max(Int(numprintedvalues), 0)
    if Int(offset) > 0
        n_below = 1 + npv
        for _ in 1:n_below
            print(io, "\e[B\r\e[2K")
        end
        print(io, "\e[A"^n_below)
    else
        print(io, "\r\e[2K")
        for _ in 1:npv
            print(io, "\e[B\r\e[2K")
        end
        npv > 0 && print(io, "\e[A"^npv)
        print(io, "\r\e[2K")
    end
    flush(io)
    return nothing
end

function _erase_spinner_line!(io::IO)
    print(io, "\r\e[2K")
    flush(io)
    return nothing
end

# Process helper paints on /dev/tty while ProgressMeter paints on stderr; clear both
# so durable logs never start on a leftover spinner fragment.
function _force_clear_spinner_line!(io::IO)
    _erase_spinner_line!(io)
    try
        if isfile("/dev/tty")
            open("/dev/tty", "w") do tty
                print(tty, "\r\e[2K")
                flush(tty)
            end
        end
    catch
    end
    return nothing
end

function _format_spinner_elapsed(t0::Float64)
    elapsed = max(0, round(Int, time() - t0))
    m, s = divrem(elapsed, 60)
    h, m = divrem(m, 60)
    if h > 0
        return string(h, ':', lpad(m, 2, '0'), ':', lpad(s, 2, '0'))
    end
    return string("0:", lpad(m, 2, '0'), ':', lpad(s, 2, '0'))
end

function _paint_spinner_line!(io::IO, desc::AbstractString, frame::Integer, t0::Float64)
    glyph = _SPINNER_GLYPHS[(Int(frame) % length(_SPINNER_GLYPHS)) + 1]
    msg = string(glyph, ' ', desc, "    Time: ", _format_spinner_elapsed(t0))
    print(io, '\r')
    printstyled(io, msg; color=_SPINNER_COLOR)
    print(io, "\e[K")
    flush(io)
    return nothing
end

function _stop_process_spinner!(reporter::_ProgressReporter)
    ps = reporter.process_spinner
    ps === nothing && return nothing
    # Pause first so the helper cannot emit another glyph frame after we decide to
    # tear down, then drop the run flag and wait for its own line clear.
    try
        touch(ps.pause_file)
    catch
    end
    try
        isfile(ps.run_flag) && rm(ps.run_flag; force=true)
    catch
    end
    try
        wait(ps.proc)
    catch
    end
    try
        rm(ps.dir; force=true, recursive=true)
    catch
    end
    reporter.process_spinner = nothing
    return nothing
end

function _stop_spinner_task!(reporter::_ProgressReporter)
    running = reporter.spinner_running
    if running !== nothing
        running[] = false
    end
    task = reporter.spinner_task
    if task !== nothing && !istaskdone(task)
        wait(task)
    end
    _stop_process_spinner!(reporter)
    reporter.spinner_running = nothing
    reporter.spinner_task = nothing
    return nothing
end

function _clear_bar!(reporter::_ProgressReporter)
    lock(reporter.io_lock) do
        bar = reporter.bar
        bar === nothing && return nothing
        npv = try
            Int(bar.numprintedvalues)
        catch
            0
        end
        offset = Int(bar.offset)
        # Skip ProgressMeter finish when showvalues lines are present: finishing with
        # empty showvalues while numprintedvalues>0 overshoots the header cursor.
        # Erase the live block from the post-update cursor position instead.
        if npv == 0
            _finish_bar!(bar)
            _erase_progress_block!(reporter.output, offset, 0)
        else
            _erase_progress_block!(reporter.output, offset, npv)
        end
        reporter.bar = nothing
        ps = reporter.process_spinner
        if ps !== nothing
            try
                isfile(ps.pause_file) && rm(ps.pause_file; force=true)
            catch
            end
        end
        # When a process helper owns the header, do not parent-repaint: concurrent
        # /dev/tty + stderr writes leave concatenated glyph residue.
        if reporter.spinner_active && ps === nothing
            _paint_spinner_line!(
                reporter.output, reporter.spinner_desc, reporter.spinner_frame, reporter.spinner_t0,
            )
        end
        return nothing
    end
end

function _clear_stage!(reporter::_ProgressReporter)
    _stop_spinner_task!(reporter)
    lock(reporter.io_lock) do
        if reporter.bar !== nothing
            npv = try
                Int(reporter.bar.numprintedvalues)
            catch
                0
            end
            offset = Int(reporter.bar.offset)
            if npv == 0
                _finish_bar!(reporter.bar)
                _erase_progress_block!(reporter.output, offset, 0)
            else
                _erase_progress_block!(reporter.output, offset, npv)
            end
            reporter.bar = nothing
        end
        if reporter.spinner_active
            _force_clear_spinner_line!(reporter.output)
            reporter.spinner_active = false
            reporter.spinner_desc = ""
            reporter.spinner_frame = 0
        else
            # Helper may have painted after the last parent erase; force-clear anyway.
            _force_clear_spinner_line!(reporter.output)
        end
    end
    _restore_blas_threads!(reporter)
    return nothing
end

_clear_stage!(::Nothing) = nothing

function _start_bar!(reporter::_ProgressReporter, description::AbstractString, total::Integer)
    reporter.enabled || return nothing
    _clear_bar!(reporter)
    lock(reporter.io_lock) do
        # Keep the process helper animating during barred stages. Pausing it made
        # evolve/TEBD headers look frozen: parent only advances the glyph on each
        # update!, and ITensor/BLAS work between updates does not yield.
        if reporter.spinner_active
            reporter.spinner_desc = String(description)
            ps = reporter.process_spinner
            if ps !== nothing
                try
                    write(ps.desc_file, reporter.spinner_desc)
                    isfile(ps.pause_file) && rm(ps.pause_file; force=true)
                catch
                end
            else
                _paint_spinner_line!(
                    reporter.output, reporter.spinner_desc, reporter.spinner_frame, reporter.spinner_t0,
                )
            end
        end
        offset = reporter.spinner_active ? 1 : 0
        reporter.bar = ProgressMeter.Progress(
            total;
            desc="$description ",
            dt=0.05,
            output=reporter.output,
            enabled=true,
            offset=offset,
            color=_BAR_COLOR,
        )
    end
    return nothing
end

function _start_julia_spinner_task!(reporter::_ProgressReporter, running::Base.RefValue{Bool})
    reporter.spinner_task = Threads.@spawn begin
        while running[]
            lock(reporter.io_lock) do
                running[] || return nothing
                reporter.spinner_active || return nothing
                # Keep animating the header even while a child ProgressMeter bar is
                # active; ProgressMeter offset=1 returns the cursor to that line.
                _paint_spinner_line!(
                    reporter.output,
                    reporter.spinner_desc,
                    reporter.spinner_frame,
                    reporter.spinner_t0,
                )
                reporter.spinner_frame += 1
                return nothing
            end
            sleep(0.08)
        end
    end
    return nothing
end

# Helper process animates the glyph while the parent is blocked in BLAS.
# Writes to /dev/tty when possible so the line stays visible on the real terminal.
function _start_process_spinner!(reporter::_ProgressReporter, description::AbstractString)
    dir = mktempdir(prefix="pt_spin_")
    run_flag = joinpath(dir, "running")
    desc_file = joinpath(dir, "desc.txt")
    pause_file = joinpath(dir, "pause")
    write(desc_file, description)
    touch(run_flag)

    python = Sys.which("python3")
    python === nothing && return nothing

    script = """
import os, sys, time
glyphs = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"] #["◐", "◓", "◑", "◒"]
run_flag, desc_file, pause_file = sys.argv[1:4]
out = sys.stderr
try:
    out = open("/dev/tty", "w", encoding="utf-8", buffering=1)
except Exception:
    out = sys.stderr
i = 0
t0 = time.time()
while os.path.exists(run_flag):
    if not os.path.exists(pause_file):
        try:
            with open(desc_file, "r", encoding="utf-8") as f:
                desc = f.read().rstrip()
        except Exception:
            desc = ""
        elapsed = max(0, int(time.time() - t0))
        m, s = divmod(elapsed, 60)
        g = glyphs[i % 4]
        out.write(f"\\r\\033[32m{g} {desc}    Time: 0:{m:02d}:{s:02d}\\033[0m\\033[K")
        out.flush()
        i += 1
    time.sleep(0.08)
try:
    out.write("\\r\\033[2K")
    out.flush()
except Exception:
    pass
"""
    cmd = `$(python) -c $script $run_flag $desc_file $pause_file`
    proc = run(cmd; wait=false)
    reporter.process_spinner = _ProcessSpinner(dir, run_flag, desc_file, pause_file, proc)
    return nothing
end

function _start_spinner!(reporter::_ProgressReporter, description::AbstractString)
    _clear_stage!(reporter)
    reporter.enabled || return nothing
    _throttle_blas_for_spinner!(reporter)
    desc = String(description)
    t0 = time()
    reporter.spinner_active = true
    reporter.spinner_desc = desc
    reporter.spinner_t0 = t0
    reporter.spinner_frame = 0
    use_process = Threads.nthreads() == 1 &&
        (reporter.output isa Base.TTY || reporter.output === stderr) &&
        Sys.which("python3") !== nothing

    # Always paint immediately from the parent so the caption is never blank.
    lock(reporter.io_lock) do
        _paint_spinner_line!(reporter.output, desc, 0, t0)
    end

    running = Ref(true)
    reporter.spinner_running = running
    if Threads.nthreads() > 1
        _start_julia_spinner_task!(reporter, running)
    elseif use_process
        _start_process_spinner!(reporter, desc)
    else
        reporter.spinner_task = @async begin
            while running[]
                lock(reporter.io_lock) do
                    running[] || return nothing
                    reporter.spinner_active || return nothing
                    _paint_spinner_line!(
                        reporter.output,
                        reporter.spinner_desc,
                        reporter.spinner_frame,
                        reporter.spinner_t0,
                    )
                    reporter.spinner_frame += 1
                    return nothing
                end
                sleep(0.08)
            end
        end
    end
    return nothing
end

function _set_spinner_desc!(reporter::_ProgressReporter, description::AbstractString)
    reporter.enabled || return nothing
    reporter.spinner_active || return nothing
    desc = String(description)
    reporter.spinner_desc = desc
    ps = reporter.process_spinner
    if ps !== nothing
        try
            write(ps.desc_file, desc)
        catch
        end
        # Helper process owns header painting; parent paints would race on /dev/tty.
        return nothing
    end
    lock(reporter.io_lock) do
        if reporter.bar === nothing
            _paint_spinner_line!(
                reporter.output, desc, reporter.spinner_frame, reporter.spinner_t0,
            )
        end
        return nothing
    end
end

_set_spinner_desc!(::Nothing, ::AbstractString) = nothing

function _ensure_spinner!(reporter::_ProgressReporter, description::AbstractString)
    reporter.enabled || return nothing
    alive = reporter.spinner_active && (
        (reporter.spinner_running !== nothing && reporter.spinner_running[]) ||
        reporter.process_spinner !== nothing
    )
    if alive
        _set_spinner_desc!(reporter, description)
        return nothing
    end
    _start_spinner!(reporter, description)
    return nothing
end

_ensure_spinner!(::Nothing, ::AbstractString) = nothing

function _update_bar!(
    reporter::_ProgressReporter,
    position::Integer;
    showvalues=(),
)
    lock(reporter.io_lock) do
        bar = reporter.bar
        bar isa ProgressMeter.Progress || return nothing
        ProgressMeter.update!(
            bar,
            Int(position);
            showvalues=_resolve_showvalues(showvalues),
            keep=false,
            force=true,
            color=_BAR_COLOR,
        )
        # ProgressMeter offset=1 leaves the cursor on the header line. Advance the
        # spinner there only when no process helper owns that line; otherwise the
        # helper animates independently and dual writers leave glyph residue.
        if reporter.spinner_active && reporter.process_spinner === nothing
            _paint_spinner_line!(
                reporter.output,
                reporter.spinner_desc,
                reporter.spinner_frame,
                reporter.spinner_t0,
            )
            reporter.spinner_frame += 1
        end
        return nothing
    end
end

function _pulse_spinner!(reporter::_ProgressReporter; showvalues=())
    lock(reporter.io_lock) do
        reporter.spinner_active || return nothing
        reporter.bar === nothing || return nothing
        _paint_spinner_line!(
            reporter.output, reporter.spinner_desc, reporter.spinner_frame, reporter.spinner_t0,
        )
        reporter.spinner_frame += 1
        return nothing
    end
end

function _with_progress(
    f::Function,
    reporter::_ProgressReporter,
    description::AbstractString,
    total::Integer,
)
    if reporter.spinner_active
        _set_spinner_desc!(reporter, description)
    end
    _start_bar!(reporter, description, total)
    update! = (position; showvalues=()) -> _update_bar!(reporter, position; showvalues)
    try
        return f(update!)
    finally
        _clear_bar!(reporter)
    end
end

function _with_spinner(
    f::Function,
    reporter::_ProgressReporter,
    description::AbstractString,
)
    _start_spinner!(reporter, description)
    try
        return f()
    finally
        _clear_stage!(reporter)
    end
end
