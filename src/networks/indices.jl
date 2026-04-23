# src/networks/indices.jl

import ITensorMPS: siteinds, siteind, linkinds, linkind, linkdim, linkdims, maxlinkdim,
                   common_siteind, common_siteinds, unique_siteind, unique_siteinds,
                   findfirstsiteind, findfirstsiteinds, findsite, findsites, firstsiteind, firstsiteinds,
                   replace_siteinds, replace_siteinds!, hassameinds, totalqn, replaceprime

# Single-arg query forwarding (return indices / integers / booleans)
# Use concrete argument types to avoid ambiguity with ITensorMPS methods
siteinds(m::AbstractMPS) = siteinds(m.core)

siteind(m::AbstractMPS, j::Integer) = siteind(m.core, j)

linkinds(m::AbstractMPS) = linkinds(m.core)

linkind(m::AbstractMPS, args...) = linkind(m.core, args...)
linkind(m::AbstractMPS, j::Integer) = linkind(m.core, j)

linkdim(m::AbstractMPS, b::Integer) = linkdim(m.core, b)

linkdims(m::AbstractMPS) = linkdims(m.core)

maxlinkdim(m::AbstractMPS) = maxlinkdim(m.core)

totalqn(m::AbstractMPS) = totalqn(m.core)

findfirstsiteind(m::AbstractMPS, args...) = findfirstsiteind(m.core, args...)
findfirstsiteind(m::AbstractMPS, s::Index) = findfirstsiteind(m.core, s)

findfirstsiteinds(m::AbstractMPS, args...) = findfirstsiteinds(m.core, args...)

findsite(m::AbstractMPS, args...) = findsite(m.core, args...)
findsite(m::AbstractMPS, s::Index) = findsite(m.core, s)

findsites(m::AbstractMPS, args...) = findsites(m.core, args...)
findsites(m::AbstractMPS, s::Index) = findsites(m.core, s)

firstsiteind(m::AbstractMPS, args...) = firstsiteind(m.core, args...)
firstsiteind(m::AbstractMPS, j::Integer) = firstsiteind(m.core, j)

firstsiteinds(m::AbstractMPS, args...) = firstsiteinds(m.core, args...)

# Two-MPS-arg query forwarding
for func in (:common_siteind, :common_siteinds, :unique_siteind, :unique_siteinds, :hassameinds)
    @eval begin
        $func(m1::AbstractMPS, m2::AbstractMPS, args...; kwargs...) = $func(m1.core, m2.core, args...; kwargs...)
        $func(m1::AbstractMPS, m2, args...; kwargs...) = $func(m1.core, m2, args...; kwargs...)
        
        # Specific overrides for ambiguities with ITensorMPS patterns
        $func(m1::AbstractMPS, m2::CoreAbstractMPS, j::Integer) = $func(m1.core, m2, j)
        $func(m1::AbstractMPS, m2::AbstractMPS, j::Integer) = $func(m1.core, m2.core, j)
        $func(m1::AbstractMPS, m2::CoreAbstractMPS) = $func(m1.core, m2)
    end
end

# Out-of-place (returns new MPS/MPO)
for func in (:replace_siteinds, :replaceprime)
    @eval $func(m::AbstractMPS, args...; kwargs...) = _rewrap(m, $func(m.core, args...; kwargs...))
end

# In-place mutating
replace_siteinds!(m::AbstractMPS, args...; kwargs...) = (replace_siteinds!(m.core, args...; kwargs...); m)

# Index tag query functions
tag_tokens(s::Index) = string.(tags(s))

has_tag_token(s::Index, token::AbstractString) = any(==(token), tag_tokens(s))

has_tag_prefix(s::Index, prefix::AbstractString) = any(t -> startswith(t, prefix), tag_tokens(s))

function tag_value(s::Index, prefix::AbstractString)
    for token in tag_tokens(s)
        startswith(token, prefix) || continue
        return String(token[length(prefix) + 1:end])
    end
    return nothing
end
