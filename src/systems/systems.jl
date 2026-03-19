# This module exports the classes that will be used to represent the systems in this package.

module Systems

struct System
    state::Any
    hamiltonian::Any
    dissipators::Any
end

System(; kwargs...) = System(nothing, nothing, nothing, nothing, (; kwargs...))

function validate_system(args...)
    nothing
end

end # module
