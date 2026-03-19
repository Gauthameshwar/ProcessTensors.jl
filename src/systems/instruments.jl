# Dispatch protocols for system-level operations

abstract type AbstractInstrument end

struct ProjectiveMeasurement <: AbstractInstrument
    operators::Any
    metadata::NamedTuple
end

struct ExpectationValueMeasurement <: AbstractInstrument
    observable::Any
    metadata::NamedTuple
end

struct CollapseOperator <: AbstractInstrument
    operator::Any
    metadata::NamedTuple
end

struct POVM <: AbstractInstrument
    operators::Any
    metadata::NamedTuple
end

function apply_instrument(system, instrument, state)
    nothing
end

function measure(system, instrument, state)
    nothing
end

function evolve_step(system, generator, state, dt)
    nothing
end

function povm_outcomes(system, instrument, state)
    nothing
end
