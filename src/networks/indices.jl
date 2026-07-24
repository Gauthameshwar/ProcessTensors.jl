# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: src/networks/indices.jl
# Contributor: Gauthameshwar S.
#
# Provides compact site and link index helpers that forward to ITensorMPS on
# wrapped MPS/MPO cores.

import ITensorMPS: siteinds, siteind, linkinds, linkind, linkdim, linkdims, maxlinkdim

siteinds(m::AbstractMPS; kwargs...) = siteinds(m.core; kwargs...)
siteinds(m::AbstractMPS, j::Integer; kwargs...) = siteinds(m.core, j; kwargs...)
siteind(m::AbstractMPS, j::Integer; kwargs...) = siteind(m.core, j; kwargs...)
linkinds(m::AbstractMPS; kwargs...) = linkinds(m.core; kwargs...)
linkind(m::AbstractMPS, j::Integer; kwargs...) = linkind(m.core, j; kwargs...)
linkdim(m::AbstractMPS, b::Integer; kwargs...) = linkdim(m.core, b; kwargs...)
linkdims(m::AbstractMPS; kwargs...) = linkdims(m.core; kwargs...)
maxlinkdim(m::AbstractMPS; kwargs...) = maxlinkdim(m.core; kwargs...)

"""
    tag_tokens(s::Index)

Return index tags as strings. ProcessTensors uses tokens such as `"Liouv"`,
`"ptype=..."`, and `"tstep=..."` for Liouville sites and process-tensor legs.
"""
tag_tokens(s::Index) = string.(tags(s))

"""Return `true` if `s` has tag `token` exactly."""
has_tag_token(s::Index, token::AbstractString) = any(==(token), tag_tokens(s))

"""Return `true` if any tag on `s` starts with `prefix`."""
has_tag_prefix(s::Index, prefix::AbstractString) = any(t -> startswith(t, prefix), tag_tokens(s))

"""
    tag_value(s::Index, prefix)

Return the suffix after `prefix` on the first matching tag, e.g. `tag_value(s, "tstep=")`.
"""
function tag_value(s::Index, prefix::AbstractString)
    for token in tag_tokens(s)
        startswith(token, prefix) || continue
        return String(token[length(prefix) + 1:end])
    end
    return nothing
end
