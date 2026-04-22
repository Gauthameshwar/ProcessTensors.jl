# src/systems/instruments.jl

module Instruments

export AbstractInstrument
export StatePreparation, ObservableMeasurement, MeasureAndDiscard, IdentityOperation, SystemPropagation

abstract type AbstractInstrument end

# 1. State Preparation (Boundary Condition at t=0)
struct StatePreparation{M<:AbstractMPS} <: AbstractInstrument
    # Can be a vector (pure state) or a matrix (density matrix)
    state::M
end

# 2. Observable Measurement (Post-selection / Projection)
struct ObservableMeasurement{O<:OpSum} <: AbstractInstrument
    # The operator you are projecting onto (e.g., [1 0; 0 0] for UP)
    Obs:O
end

# 3. Measure and Discard (The Trace Operation)
struct MeasureAndDiscard <: AbstractInstrument end # Needs NO fields! It is universally the trace operation (Identity matrix).


# 4. "Doing Nothing"
struct IdentityOperation <: AbstractInstrument
    # No fields needed. Tells the PT to just propagate forward.
end

# 5. Free Evolution of the System
struct SystemPropagation{S<:AbstractSystem} <: AbstractInstrument
    # The system propagator is the e^Lt operator
    Sys::S
end

end # module