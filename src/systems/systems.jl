# src/systems/systems.jl

abstract type AbstractSystem end

struct SpinSystem <: AbstractSystem
    H::OpSum
    Ls::Vector{OpSum}
end

struct BosonSystem <: AbstractSystem
    H::OpSum
    Ls::Vector{OpSum}
end
