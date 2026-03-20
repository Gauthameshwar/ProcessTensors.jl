# src/mpo/mpo_utils.jl

# Import the functions from ITensorMPS that we want to extend
import ITensorMPS: random_mpo

random_mpo(args...; kwargs...) = MPO(ITensorMPS.random_mpo(args...; kwargs...))
