# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: scripts/terminal/tebd.jl
# Contributor: Gauthameshwar S.
#
# Demonstrates TEBD gate preparation, live evolution progress, and persistent
# verbose logging on a short two-spin chain.
#
# Run with:
# julia --project=. scripts/terminal/tebd.jl

using ProcessTensors
using ITensors
using ITensors.Ops: Trotter

progress = true
verbose = true
dt = 0.05
T = 20.0

@info "Defining the system and Hamiltonian..."
N = 12
sites = siteinds("S=1/2", N)

H_lazy = let
    h = OpSum()
    for i in 1:N
        h += (0.8, "Sx", i)
    end
    for i in 1:N-1
        h += (0.6, "Sz", i, "Sz", i+1)
        h += (0.4, "Sy", i, "Sy", i+1)
        h += (0.2, "Sx", i, "Sx", i+1)
    end
    h
end

psi0 = MPS(sites, fill("Up", N))

@info "Running TEBD..."
psiT = tebd(
    psi0,
    H_lazy,
    dt,
    T;
    alg=Trotter{4}(),
    maxdim=32,
    cutoff=1e-10,
    progress=progress,
    verbose=verbose,
)

println("Final TEBD max bond dimension: ", maxlinkdim(psiT))
