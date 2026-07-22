# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: src/time_evolution/tebd.jl
# Contributor: Gauthameshwar S.
#
# Implements TEBD-style time evolution routines used to propagate tensor-network
# states and operators.

import ITensors: ITensor, Index, exp, replaceind
import ITensors: exp as itensor_exp
import ITensors.Ops: Exact, Trotter
import ITensorMPS: OpSum, apply

"""
    propagator_itensor_from_gates(gates, sites)

Contract a vector of Trotter gates into a single superoperator `ITensor` on
`sites` by successive gate contraction with index promotion.

Returns `(U::ITensor, final_out::Vector{Index})` where `U` has unprimed `sites`
as ket/output legs and `final_out` as bra/input legs.
"""
function propagator_itensor_from_gates(
    gates::AbstractVector{<:ITensor},
    sites::AbstractVector{<:Index},
)
    current_out = Dict(s => prime(s) for s in sites)

    U = ITensor(1.0)
    for s in sites
        U *= delta(current_out[s], s)
    end

    for gate in gates
        g = gate
        next_out = copy(current_out)

        for s in sites
            if hasind(g, s)
                g = replaceind(g, s, current_out[s])
            end

            sp = prime(s)
            if hasind(g, sp)
                promoted = prime(current_out[s])
                g = replaceind(g, sp, promoted)
                next_out[s] = promoted
            end
        end

        U = g * U
        current_out = next_out
    end

    final_out = Index[current_out[s] for s in sites]
    return U, final_out
end

"""
    liouvillian_propagator(os, sites, dt; alg=Exact(), jump_ops=[], liouville_form=false)

Build the one-step Liouville propagator ``U = \\exp(dt \\, L)`` as a single
`ITensor` on `sites`.

When `liouville_form=false`, `os` is a physical Hamiltonian and `L` is built by
[`liouvillian_opsum`](@ref):

```math
L = -i[H,\\cdot] + \\sum_k \\gamma_k \\, \\mathcal{D}[L_k],
\\qquad
\\mathcal{D}[L]\\rho = L\\rho L^\\dagger - \\tfrac{1}{2}\\{L^\\dagger L, \\rho\\}.
```

Set `liouville_form=true` when `os` is already a Liouville `OpSum`.

`alg` selects how ``\\exp(dt\\,L)`` is constructed:

- `Exact()` (default): contract `L` to a dense superoperator matrix and compute
  ``U = \\exp(dt\\,L)`` exactly. Suitable for small Liouville dimensions.
- `Trotter{n}()`: approximate
  ``\\exp(dt\\,L) \\approx \\prod_j \\exp(dt\\,L_j)`` using [`trotter_gates`](@ref)
  (orders `1`, `2`, and even `n >= 4`), then contract the gate list with
  [`propagator_itensor_from_gates`](@ref).

Leg convention: unprimed `sites` are ket/output legs; `prime.(sites)` are
bra/input legs.
"""
function liouvillian_propagator(
    os::OpSum,
    sites::AbstractVector{<:Index},
    dt::Real;
    alg=Exact(),
    jump_ops=Tuple{Number,String,Int}[],
    liouville_form::Bool=false,
)
    L = liouville_form ? os : liouvillian_opsum(os, jump_ops)

    if alg isa Exact
        L_mpo = ITensorMPS.MPO(L, sites)
        L_dense = foldl(*, L_mpo)
        return itensor_exp(dt * L_dense)
    elseif alg isa Trotter
        gates = trotter_gates(L, sites, dt; alg=alg)
        U, final_out = propagator_itensor_from_gates(gates, sites)

        for (old_out, s) in zip(final_out, sites)
            U = replaceind(U, old_out, prime(s))
        end

        return U
    end

    throw(ArgumentError("Unsupported exponentiation algorithm: $(typeof(alg))."))
end

# Deprecated compatibility alias; remove in a later 0.y release after the migration window.
Base.@deprecate liouvillian_propagator_itensor(args...; kwargs...) liouvillian_propagator(args...; kwargs...)

function _tebd_loop(
    state::AbstractMPS,
    gates::Vector{ITensor},
    dt::Real,
    T::Real;
    maxdim::Int=typemax(Int),
    cutoff::Real=1e-8,
    verbose::Bool=false,
)
    function nsteps(T_duration::Real, dt_step::Real; atol=1e-12, rtol=1e-12)
        dt_step > 0 || throw(ArgumentError("dt must be positive; got dt=$dt_step."))
        T_duration ≥ 0 || throw(ArgumentError("T must be nonnegative; got T=$T_duration."))
        n = round(Int, T_duration / dt_step)
        isapprox(n * dt_step, T_duration; atol=atol, rtol=rtol) ||
            throw(ArgumentError("T/dt must be approximately integer. Got T=$T_duration, dt=$dt_step."))
        return n
    end

    N_steps = nsteps(T, dt)
    ψ = copy(state)
    for step in 1:N_steps
        ψ = apply(gates, ψ; cutoff=cutoff, maxdim=maxdim)
        if verbose
            χ_max = maxlinkdim(ψ)
            println("TEBD step $step / $N_steps  |  max bond dim = $χ_max")
        end
    end
    return ψ
end

"""
    tebd(state::AbstractMPS{Hilbert}, H, dt, T; alg=Trotter{2}(), maxdim, cutoff, verbose)
    tebd(state::AbstractMPS{Liouville}, H, dt, T; jump_ops=[], alg=Trotter{2}(), maxdim, cutoff, verbose)

Time-evolve an MPS for total time `T` in steps of `dt` using TEBD.

For `MPS{Hilbert}`, each step applies ``U = \\exp(-i H\\, dt)``.
For `MPS{Liouville}`, the Liouvillian ``L`` is built internally and each step
applies ``U = \\exp(L\\, dt)``.

`alg` must be a `Trotter{n}()` object from `ITensors.Ops`. It controls the
Suzuki–Trotter factorization of the per-step propagator via
[`trotter_gates`](@ref); supported orders are `1`, `2`, and even `n >= 4`.
`Trotter{2}()` is the default second-order choice.
The gates are applied `round(T/dt)` times with optional truncation (`maxdim`,
`cutoff`).

For exact single-step exponentiation on small Liouville spaces, use
[`liouvillian_propagator`](@ref) with `alg=Exact()` instead of TEBD.

# Examples
```julia
ψ_T = tebd(ψ0, H, 0.1, 1.0; alg=Trotter{2}())
ρL_T = tebd(ρL0, H, 0.1, 1.0; jump_ops=[(0.1, "S-", 1)])
```
"""
function tebd(
    state::AbstractMPS{Hilbert},
    H::OpSum,
    dt::Real,
    T::Real;
    alg=Trotter{2}(),
    maxdim::Int=typemax(Int),
    cutoff::Real=1e-8,
    verbose::Bool=false,
)
    gates = trotter_gates(H, siteinds(state), -im * dt; alg=alg)
    return _tebd_loop(state, gates, dt, T; maxdim, cutoff, verbose)
end

"""
    tebd(state::AbstractMPS{Liouville}, H, dt, T; jump_ops=[], alg=Trotter{2}(), maxdim, cutoff, verbose)

Liouville-space overload: see the main [`tebd`](@ref) docstring.
"""
function tebd(
    state::AbstractMPS{Liouville},
    H::OpSum,
    dt::Real,
    T::Real;
    jump_ops=Tuple{Number,String,Int}[],
    alg=Trotter{2}(),
    maxdim::Int=typemax(Int),
    cutoff::Real=1e-8,
    verbose::Bool=false,
)
    L = liouvillian_opsum(H, jump_ops)
    gates = trotter_gates(L, siteinds(state), dt; alg=alg)
    return _tebd_loop(state, gates, dt, T; maxdim, cutoff, verbose)
end
