# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: src/terminal/progressmeter_backend.jl
# Contributor: Gauthameshwar S.
#
# Isolates every ProgressMeter.jl dependency: deterministic-bar creation,
# throttled updates with compact inline values, and bar-only erasure.

import ProgressMeter
using Printf: @sprintf

const _BAR_COLOR = :cyan
const _BAR_REFRESH_INTERVAL = 0.05

"""
    _ProgressMeterBar

Bar state backed by one `ProgressMeter.Progress` pinned one line below the
spinner header (`offset=1`). Tracked values are rendered inline into the bar
description so the bar always occupies exactly one terminal line
(`numprintedvalues` stays zero and never needs cursor workarounds).
"""
mutable struct _ProgressMeterBar <: _AbstractBarState
    meter::ProgressMeter.Progress
    total::Int
    description::String
end

# Compact one-line rendering of tracked values, e.g. "  t=2.15  χ=128".
# Floats and complexes always use exactly two decimal digits so the bar
# description length stays stable (avoids the bar "walking" left/right).
# These values are display-only and never feed back into the calculation.
_format_bar_value(v) = string(v)
_format_bar_value(v::AbstractFloat) = @sprintf("%.2f", v)
function _format_bar_value(v::Complex)
    im = imag(v)
    return string(
        @sprintf("%.2f", real(v)),
        im < 0 ? "-" : "+",
        @sprintf("%.2f", abs(im)),
        "im",
    )
end

function _format_bar_values(values)
    isempty(values) && return ""
    return string(
        "  ",
        join((string(k, "=", _format_bar_value(v)) for (k, v) in values), "  "),
    )
end

function _create_bar(io::IO, description::AbstractString, total::Integer)
    meter = ProgressMeter.Progress(
        Int(total);
        desc=string(description, ' '),
        dt=_BAR_REFRESH_INTERVAL,
        output=io,
        enabled=true,
        offset=1,
        color=_BAR_COLOR,
    )
    bar = _ProgressMeterBar(meter, Int(total), String(description))
    # Paint the empty bar immediately so the two-line layout appears at once.
    ProgressMeter.update!(meter, 0; force=true)
    return bar
end

# Advance the bar. ProgressMeter throttles physical repaints internally through
# its `dt`; the final position is always painted. With `offset=1` every repaint
# returns the cursor to the header line, preserving the display invariant.
function _update_bar!(bar::_ProgressMeterBar, position::Integer; values=())
    desc = string(bar.description, _format_bar_values(values), ' ')
    ProgressMeter.update!(
        bar.meter,
        Int(position);
        desc=desc,
        force=Int(position) >= bar.total,
    )
    return nothing
end

# Repaint the bar at its current position (used after persistent logs interrupt
# the transient display).
function _repaint_bar!(bar::_ProgressMeterBar)
    ProgressMeter.update!(bar.meter; force=true)
    return nothing
end

_bar_position(bar::_ProgressMeterBar) = bar.meter.counter

# Erase only the bar line, leaving the spinner header. The cursor rests on the
# header line both before and after.
function _erase_bar!(io::IO, ::_ProgressMeterBar)
    _erase_line_below!(io)
    return nothing
end
