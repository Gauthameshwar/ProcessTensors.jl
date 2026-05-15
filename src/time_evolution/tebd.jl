# src/time_evolution/tebd.jl

import ITensors: op, apply, exp, ops, ITensor, replaceind
import ITensors: exp as itensor_exp
import ITensors.Ops: Exact, Trotter, Prod, Sum
import ITensorMPS: OpSum

# --------------------- Space Detection Helpers ---------------------

# Check whether a ProcessTensors state lives in Liouville space
is_liouville_space(::AbstractMPS{Liouville}) = true
is_liouville_space(::AbstractMPS{Hilbert})    = false

#Check whether a ProcessTensors state lives in Hilbert space
is_hilbert_space(::AbstractMPS{Hilbert})    = true
is_hilbert_space(::AbstractMPS{Liouville})  = false

# --------------------- Gate Construction ---------------------

"""
    _build_trotter_gates(os::OpSum, sites, dt; order=2)

Build Trotter gates from a symbolic OpSum, a site vector, and a time step.

For Hilbert space the propagator is `exp(-i H dt)`, so we pass `-im * dt * os`.
For Liouville space the propagator is `exp(L dt)`, so we pass `dt * os`
(the `-im` and `gamma` factors are already baked into the Liouvillian OpSum by
`build_liouvillian_opsum`).

Returns a `Vector{ITensor}` of gates ready for `apply`.
"""
function _build_trotter_gates(os::OpSum, sites::AbstractVector{<:Index}, dt::Number; order::Int=2)
    lazy_gates = exp(dt * os; alg=Trotter{order}())
    # Materialize the lazy Trotter product into concrete ITensor gates
    gate_prod = Prod{ITensor}(lazy_gates, collect(sites))
    # Extract the individual gate ITensors from the Prod wrapper
    return collect(ITensor, only(gate_prod.args))
end

"""Compose Trotter gates into a single propagation ITensor on Liouville sites."""
function _compose_gates_to_map(gates::AbstractVector{<:ITensor}, base_sites::AbstractVector{<:Index})
    curr_out = Dict{Index,Index}(s => prime(s) for s in base_sites)
    map_t = ITensor(1.0)
    for s in base_sites
        map_t *= delta(curr_out[s], s)
    end

    for gate in gates
        g2 = gate
        next_out = copy(curr_out)
        for s in base_sites
            hasind(g2, s) && (g2 = replaceind(g2, s, curr_out[s]))
            sp = prime(s)
            if hasind(g2, sp)
                promoted = prime(curr_out[s])
                g2 = replaceind(g2, sp, promoted)
                next_out[s] = promoted
            end
        end
        map_t = g2 * map_t
        curr_out = next_out
    end

    final_out = Index[curr_out[s] for s in base_sites]
    return map_t, final_out
end

"""
    liouvillian_propagator_itensor(os::OpSum, sites, dt; exp_alg=Exact(), jump_ops=[], liouville_form=false)

Build `U = exp(dt * L)` as an ITensor on Liouville `sites`. By default `os` is a **physical**
Hamiltonian and the Liouvillian is formed via `OpSum_Liouville(os; jump_ops)`. Set
`liouville_form=true` when `os` is already a Liouvillian `OpSum`.

`exp_alg` selects the exponentiation algorithm:

- `Exact()`: contract `MPO(L)` to a single operator ITensor and apply `exp(dt * L)` exactly.
- `Trotter{n}()`: lazy Trotter product from `exp(dt * L; alg=Trotter{n}())`, composed to one map.

Leg convention matches `ITensor(exp(dt * L_mpo))`: unprimed `sites` are ket/output legs,
`prime.(sites)` are bra/input legs.
"""
function liouvillian_propagator_itensor(
    os::OpSum,
    sites::AbstractVector{<:Index},
    dt::Real;
    exp_alg=Exact(),
    jump_ops=Tuple{Number,String,Int}[],
    liouville_form::Bool=false,
)
    L_os = liouville_form ? os : OpSum_Liouville(os, jump_ops)
    if exp_alg isa Exact
        L_mpo = ITensorMPS.MPO(L_os, sites)
        Lj = foldl(*, L_mpo)
        return itensor_exp(dt * Lj)
    elseif exp_alg isa Trotter
        lazy_gates = exp(dt * L_os; alg=exp_alg)
        gate_prod = Prod{ITensor}(lazy_gates, collect(sites))
        gates = collect(ITensor, only(gate_prod.args))
        U_map, final_out = _compose_gates_to_map(gates, sites)
        for (s, s_in) in zip(sites, final_out)
            U_map = replaceind(U_map, s_in, prime(s))
        end
        return U_map
    end
    throw(ArgumentError("liouvillian_propagator_itensor: unsupported exp_alg=$(typeof(exp_alg))."))
end

# --------------------- Main TEBD Entry Points ---------------------

"""
    tebd(state::AbstractMPS{Hilbert}, os_H::OpSum, dt, T; kwargs...)

Evolve a **Hilbert-space** MPS under the Hamiltonian `os_H` for total time `T`
using second-order TEBD (Trotter).

The propagator is `U(dt) = exp(-i H dt)`.

# Keywords
- `maxdim::Int = typemax(Int)`: maximum bond dimension after each gate application.
- `cutoff::Float64 = 1e-8`: SVD truncation cutoff.
- `order::Int = 2`: Trotter order (1 or 2).
- `verbose::Bool = false`: print progress every 10 steps.

Returns the time-evolved MPS (same type as input).
"""
function tebd(
    state::AbstractMPS{Hilbert},
    os_H::OpSum,
    dt::Real,
    T::Real;
    maxdim::Int=typemax(Int),
    cutoff::Float64=1e-8,
    order::Int=2,
    verbose::Bool=false,
)
    sites = siteinds(state)
    gates = _build_trotter_gates(os_H, sites, -im * dt; order=order)

    return _tebd_loop(state, gates, dt, T; maxdim=maxdim, cutoff=cutoff, verbose=verbose)
end

"""
    tebd(state::AbstractMPS{Liouville}, os_H::OpSum, dt, T; jump_ops=[], kwargs...)

Evolve a **Liouville-space** MPS (vectorised density matrix) under the Lindblad
master equation for total time `T` using second-order TEBD (Trotter).

The propagator is `exp(L dt)` where `L` is the full Liouvillian superoperator.

# Arguments
- `os_H::OpSum`: physical Hamiltonian in terms of standard operator names (e.g. `"Sz"`, `"S+"`) and **physical** site numbers.
- `jump_ops`: Lindblad jump operators, in any format accepted by `OpSum_Liouville`.

# Keywords
- `maxdim::Int = typemax(Int)`: maximum bond dimension.
- `cutoff::Float64 = 1e-8`: SVD truncation cutoff.
- `order::Int = 2`: Trotter order.
- `verbose::Bool = false`: print progress.

Returns the time-evolved MPS{Liouville} (same type as input).
"""
function tebd(
    state::AbstractMPS{Liouville},
    os_H::OpSum,
    dt::Real,
    T::Real;
    jump_ops=Tuple{Number,String,Int}[],
    maxdim::Int=typemax(Int),
    cutoff::Float64=1e-8,
    order::Int=2,
    verbose::Bool=false,
)
    # Build the full Liouvillian OpSum (commutator + dissipator)
    L_os = OpSum_Liouville(os_H, jump_ops)

    # The Liouville sites are the site indices of the vectorised state
    liouv_sites_vec = siteinds(state)

    # dt is REAL — the -im factors are already inside the Liouvillian OpSum
    gates = _build_trotter_gates(L_os, liouv_sites_vec, dt; order=order)

    return _tebd_loop(state, gates, dt, T; maxdim=maxdim, cutoff=cutoff, verbose=verbose)
end

# --------------------- TEBD Time Loop ---------------------

"""
    _tebd_loop(state, gates, dt, T; maxdim, cutoff, verbose)

Execute the TEBD time-stepping loop.

For `Trotter{2}`, `exp` already produces the symmetric 2nd-order decomposition:
`exp(H/2) * reverse(exp(H/2))`, so applying the full gate list once is one TEBD2 step.

For `Trotter{1}`, `exp` produces a 1st-order decomposition.
"""
function _tebd_loop(
    state::AbstractMPS,
    gates::Vector{ITensor},
    dt::Real,
    T::Real;
    maxdim::Int=typemax(Int),
    cutoff::Float64=1e-8,
    verbose::Bool=false,
)
    N_steps = Int(round(T / dt))

    ψ = copy(state)
    for step in 1:N_steps
        ψ = apply(gates, ψ; cutoff=cutoff, maxdim=maxdim)

        if verbose #&& (step % 10 == 0 || step == 1 || step == N_steps)
            χ_max = maxlinkdim(ψ)
            println("TEBD step $step / $N_steps  |  max bond dim = $χ_max")
        end
    end

    return ψ
end
