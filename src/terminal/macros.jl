# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: src/terminal/macros.jl
# Contributor: Gauthameshwar S.
#
# Defines the five thin contributor-facing progress macros. Each macro
# evaluates its arguments exactly once and delegates to reporting.jl.

# Accept both a NamedTuple literal `(a=1, b=2)` and the bare single assignment
# `(a=1)`, which Julia parses as an assignment expression rather than a tuple.
function _progress_meta_expr(ex)
    if ex isa Expr && ex.head === :(=) && ex.args[1] isa Symbol
        return Expr(:tuple, ex)
    end
    return ex
end

"""
    @progress_start progress verbose "Operation"
    @progress_start progress verbose "Operation" (key=value, ...)

Begin one tracked operation and return its run reporter. Call once per public
workflow; pass the returned `run` to internal implementations. Emits the
persistent start checkpoint when `verbose=true`.
"""
macro progress_start(progress, verbose, operation, meta...)
    length(meta) <= 1 ||
        error("@progress_start accepts at most one metadata tuple.")
    metaex = isempty(meta) ? :((;)) : _progress_meta_expr(meta[1])
    quote
        local run = _run_reporter($(esc(progress)), $(esc(verbose)))
        _progress_start!(run, $(esc(operation)); pairs($(esc(metaex)))...)
        run
    end
end

"""
    @progress_stage run "Stage description"
    @progress_stage run "Stage description" (key=value, ...)

Declare one substantive stage: updates the transient spinner header and emits
the persistent `@info` checkpoint when `verbose=true`.
"""
macro progress_stage(run, description, meta...)
    length(meta) <= 1 ||
        error("@progress_stage accepts at most one metadata tuple.")
    metaex = isempty(meta) ? :((;)) : _progress_meta_expr(meta[1])
    quote
        _progress_stage!($(esc(run)), $(esc(description)); pairs($(esc(metaex)))...)
    end
end

"""
    @progress_bar run "Stage description" total begin ... end

Wrap one known-length loop: creates the bar below the spinner, executes the
body, and erases the completed bar in `finally`, returning to spinner-only
display. The body's return value and exceptions are preserved, and the body
always executes even when progress is disabled.
"""
macro progress_bar(run, description, total, body)
    quote
        local run_local = $(esc(run))
        _begin_progress_bar!(run_local, $(esc(description)), $(esc(total)))
        try
            $(esc(body))
        finally
            _finish_progress_bar!(run_local)
        end
    end
end

"""
    @progress_update run position
    @progress_update run position (key=value, ...)

Advance the active bar after one completed unit of work, with at most two
compact live values rendered inline on the bar line.
"""
macro progress_update(run, position, meta...)
    length(meta) <= 1 ||
        error("@progress_update accepts at most one values tuple.")
    metaex = isempty(meta) ? :((;)) : _progress_meta_expr(meta[1])
    quote
        _progress_update!($(esc(run)), $(esc(position)); pairs($(esc(metaex)))...)
    end
end

"""
    @progress_finish run

Stop the spinner worker and remove the entire transient display. Place inside
the workflow's `finally` block; safe after exceptions and idempotent.
"""
macro progress_finish(run)
    quote
        _progress_finish!($(esc(run)))
    end
end
