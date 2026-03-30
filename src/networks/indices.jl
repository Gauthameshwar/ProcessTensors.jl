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
linkdim(m::AbstractMPS, b::Integer) = linkdim(m.core, b)
linkdims(m::AbstractMPS) = linkdims(m.core)
maxlinkdim(m::AbstractMPS) = maxlinkdim(m.core)
totalqn(m::AbstractMPS) = totalqn(m.core)

findfirstsiteind(m::AbstractMPS, args...) = findfirstsiteind(m.core, args...)
findfirstsiteinds(m::AbstractMPS, args...) = findfirstsiteinds(m.core, args...)
findsite(m::AbstractMPS, args...) = findsite(m.core, args...)
findsites(m::AbstractMPS, args...) = findsites(m.core, args...)
firstsiteind(m::AbstractMPS, args...) = firstsiteind(m.core, args...)
firstsiteinds(m::AbstractMPS, args...) = firstsiteinds(m.core, args...)

# Two-MPS-arg query forwarding
for func in (:common_siteind, :common_siteinds, :unique_siteind, :unique_siteinds, :hassameinds)
    @eval begin
        $func(m1::AbstractMPS, m2::AbstractMPS, args...; kwargs...) = $func(m1.core, m2.core, args...; kwargs...)
        $func(m1::AbstractMPS, m2, args...; kwargs...) = $func(m1.core, m2, args...; kwargs...)
    end
end

# Out-of-place (returns new MPS/MPO)
for func in (:replace_siteinds, :replaceprime)
    @eval $func(m::AbstractMPS, args...; kwargs...) = _rewrap(m, $func(m.core, args...; kwargs...))
end

# In-place mutating
replace_siteinds!(m::AbstractMPS, args...; kwargs...) = (replace_siteinds!(m.core, args...; kwargs...); m)
