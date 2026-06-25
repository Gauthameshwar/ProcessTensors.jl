# src/networks/indices.jl
# Tier C: site/link index queries and replacements forward to ITensorMPS on `.core` (see API page).

import ITensorMPS: siteinds, siteind, linkinds, linkind, linkdim, linkdims, maxlinkdim,
                   common_siteind, common_siteinds, unique_siteind, unique_siteinds,
                   findfirstsiteind, findfirstsiteinds, findsite, findsites, firstsiteind, firstsiteinds,
                   replace_siteinds, replace_siteinds!, hassameinds, totalqn, replaceprime

siteinds(m::AbstractMPS) = siteinds(m.core)
siteind(m::AbstractMPS, j::Integer) = siteind(m.core, j)
linkinds(m::AbstractMPS) = linkinds(m.core)
linkind(m::AbstractMPS, j::Integer) = linkind(m.core, j)
linkdim(m::AbstractMPS, b::Integer) = linkdim(m.core, b)
linkdims(m::AbstractMPS) = linkdims(m.core)
maxlinkdim(m::AbstractMPS) = maxlinkdim(m.core)
totalqn(m::AbstractMPS) = totalqn(m.core)
findfirstsiteind(m::AbstractMPS, s::Index) = findfirstsiteind(m.core, s)
findfirstsiteinds(m::AbstractMPS, s::Index) = findfirstsiteinds(m.core, s)
findsite(m::AbstractMPS, s::Index) = findsite(m.core, s)
findsite(m::AbstractMPS, j::Integer) = findsite(m.core, j)
findsites(m::AbstractMPS, s::Index) = findsites(m.core, s)
findsites(m::AbstractMPS, j::Integer) = findsites(m.core, j)
firstsiteind(m::AbstractMPS, j::Integer) = firstsiteind(m.core, j)
firstsiteinds(m::AbstractMPS) = firstsiteinds(m.core)

for func in (:common_siteind, :common_siteinds, :unique_siteind, :unique_siteinds, :hassameinds)
    @eval begin
        $func(m1::AbstractMPS, m2::AbstractMPS, args...; kwargs...) = $func(m1.core, m2.core, args...; kwargs...)
        $func(m1::AbstractMPS, m2::CoreAbstractMPS, args...; kwargs...) = $func(m1.core, m2, args...; kwargs...)
        $func(m1::AbstractMPS, m2::CoreAbstractMPS, j::Integer) = $func(m1.core, m2, j)
        $func(m1::AbstractMPS, m2::AbstractMPS, j::Integer) = $func(m1.core, m2.core, j)
    end
end

for func in (:replace_siteinds, :replaceprime)
    @eval $func(m::AbstractMPS, args...; kwargs...) = _rewrap(m, $func(m.core, args...; kwargs...))
end

replace_siteinds!(m::AbstractMPS, args...; kwargs...) = (replace_siteinds!(m.core, args...; kwargs...); m)

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
