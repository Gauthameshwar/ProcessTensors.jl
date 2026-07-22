# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: src/terminal/tty_renderer.jl
# Contributor: Gauthameshwar S.
#
# Implements spinner-line rendering, elapsed-time formatting, cursor
# visibility, and line-erase primitives for the terminal progress backend.

# const _SPINNER_GLYPHS = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
# const _SPINNER_GLYPHS = ("⌜", "⌝", "⌟", "⌞")
const _SPINNER_GLYPHS = ("∘∘∘∘∘", "●∘∘∘∘", "●●∘∘∘", "●●●∘∘", "●●●●∘", "●●●●●", "∘●●●●", "∘∘●●●", "∘∘∘●●", "∘∘∘∘●", "∘∘∘∘∘")
const _SPINNER_COLOR = :green
const _SPINNER_INTERVAL = 0.10

function _format_elapsed(seconds::Real)
    total = max(0, round(Int, seconds))
    m, s = divrem(total, 60)
    h, m = divrem(m, 60)
    h > 0 && return string(h, ':', lpad(m, 2, '0'), ':', lpad(s, 2, '0'))
    return string("0:", lpad(m, 2, '0'), ':', lpad(s, 2, '0'))
end

function _render_spinner_line(
    operation::AbstractString,
    description::AbstractString,
    frame::Integer,
    started::Float64,
)
    glyph = _SPINNER_GLYPHS[mod1(Int(frame) + 1, length(_SPINNER_GLYPHS))]
    header = if isempty(description) || description == operation
        operation
    else
        string(operation, " — ", description)
    end
    return string(glyph, ' ', header, "    Time: ", _format_elapsed(time() - started))
end

# Write bytes without Julia's libuv-backed print/flush path.
#
# `print`/`flush` on stderr are serviced on the main thread and can block for the
# whole duration of a BLAS foreign call (same failure mode as `Base.sleep`).
# POSIX `write(2)` does not. `IOBuffer` (unit tests) keeps the normal `write`
# path so output stays capturable.
function _raw_write!(io::IO, bytes::AbstractVector{UInt8})
    if io isa IOBuffer
        write(io, bytes)
        return nothing
    end
    fd = if io === stdout
        Cint(1)
    elseif io isa IOStream
        Cint(Base.fd(io))
    else
        Cint(2) # stderr and TTY endpoints
    end
    ccall(:write, Cssize_t, (Cint, Ptr{UInt8}, Csize_t), fd, bytes, Csize_t(length(bytes)))
    return nothing
end

_raw_write!(io::IO, s::AbstractString) = _raw_write!(io, codeunits(s))

# One write per frame: return to column 0, paint the complete line, clear the rest.
# The cursor invariant for the whole display is "resting on the header line".
function _paint_spinner_line!(
    io::IO,
    operation::AbstractString,
    description::AbstractString,
    frame::Integer,
    started::Float64,
)
    line = _render_spinner_line(operation, description, frame, started)
    _raw_write!(io, string('\r', "\e[32m", line, "\e[0m", "\e[K"))
    return nothing
end

function _erase_current_line!(io::IO)
    _raw_write!(io, "\r\e[2K")
    return nothing
end

# Erase the single line below the cursor and return to the current line.
function _erase_line_below!(io::IO)
    _raw_write!(io, "\e[B\r\e[2K\e[A\r")
    return nothing
end

function _hide_cursor!(io::IO)
    io isa Base.TTY || return false
    _raw_write!(io, "\e[?25l")
    return true
end

function _show_cursor!(io::IO)
    _raw_write!(io, "\e[?25h")
    return nothing
end
