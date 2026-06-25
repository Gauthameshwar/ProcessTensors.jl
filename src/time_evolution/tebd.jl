# src/time_evolution/tebd.jl

import ITensors: ITensor, Index, exp, replaceind
import ITensors: exp as itensor_exp
import ITensors.Ops: Exact, Trotter, Prod
import ITensorMPS: OpSum, apply as mps_apply

# Tier C: `Exact` and `Trotter` are re-exported from `ITensors.Ops` (see API page).

"""
    trotter_gates(os, sites, τ; alg=Trotter{2}())

Factorize ``\\exp(\\tau \\, \\mathrm{os})`` into a vector of ITensor gates using a
Trotter decomposition.

Schematically, for `alg = Trotter{2}()` the decomposition has the form

```math
\\exp(\\tau \\, \\mathrm{os}) \\approx
\\prod_{\\text{groups}} \\exp\\!\\left(\\frac{\\tau}{2}\\, \\mathrm{os}_{\\text{group}}\\right)
\\cdots
\\prod_{\\text{groups}} \\exp\\!\\left(\\frac{\\tau}{2}\\, \\mathrm{os}_{\\text{group}}\\right),
```

with the exact gate ordering and grouping determined by ITensors for the chosen
`Trotter{n}()` order. Higher `n` generally improves accuracy at fixed `τ`.

`τ` encodes the full exponent prefactor, including any imaginary unit and sign:
- Hilbert space: `τ = -im * dt` for unitary evolution `exp(-im H dt)`
- Liouville space: `τ = dt` (the `-im` factors are already in `OpSum_Liouville`)
"""
function trotter_gates(
    os::OpSum,
    sites::AbstractVector{<:Index},
    τ::Number;
    alg=Trotter{2}(),
)
    lazy = exp(τ * os; alg=alg)
    prod = Prod{ITensor}(lazy, collect(sites))
    return collect(ITensor, only(prod.args))
end

"""
    propagator_itensor_from_gates(gates, sites; method=:auto)

Contract a vector of Trotter gates into a single superoperator ITensor on `sites`.

`method` controls how the contraction is performed:
- `:basis` (default for 1 site): builds the map column-by-column by applying the gates to
  Liouville basis states. This matches the TEBD path exactly and is trace-preserving by
  construction.
- `:contract`: directly contracts gate ITensors using index promotion. Suitable for
  multi-site maps.
- `:auto`: selects `:basis` for 1 site, `:contract` otherwise.

Returns `(U::ITensor, final_out::Vector{Index})` where `U` has unprimed `sites` as
ket/output legs and `final_out` as bra/input legs.
"""
function propagator_itensor_from_gates(
    gates::AbstractVector{<:ITensor},
    sites::AbstractVector{<:Index};
    method::Symbol = length(sites) == 1 ? :basis : :contract,
)
    method === :basis && return _map_from_gates_by_basis(gates, sites)
    method === :contract && return _map_from_gates_by_contraction(gates, sites)

    throw(ArgumentError("Unknown map materialization method: $method"))
end

function _map_from_gates_by_basis(
    gates::AbstractVector{<:ITensor},
    sites::AbstractVector{<:Index},
)
    length(sites) == 1 ||
        throw(ArgumentError(":basis materialization currently supports only one site."))

    s = only(sites)
    d2 = dim(s)
    d = isqrt(d2)
    phys = _phys_site_from_liouv(s)

    U = zeros(ComplexF64, d2, d2)

    for j in 1:d2
        # Construct the j-th Liouville basis vector vec(E_{ij}) where j = i + (k-1)*d
        i = ((j - 1) % d) + 1
        k = div(j - 1, d) + 1
        M = zeros(ComplexF64, d, d)
        M[i, k] = 1.0
        ρ_h = MPO{Hilbert}(CoreMPO([ITensor(M, prime(phys), phys)]))
        basis_j = to_liouville(ρ_h; sites=Index[s])

        ψj = mps_apply(gates, basis_j; maxdim=typemax(Int), cutoff=0.0)

        for ii in 1:d2
            ii_row = ((ii - 1) % d) + 1
            ii_col = div(ii - 1, d) + 1
            M_ii = zeros(ComplexF64, d, d)
            M_ii[ii_row, ii_col] = 1.0
            ρ_ii = MPO{Hilbert}(CoreMPO([ITensor(M_ii, prime(phys), phys)]))
            basis_ii = to_liouville(ρ_ii; sites=Index[s])
            U[ii, j] = inner(basis_ii, ψj)
        end
    end

    return ITensor(U, s, prime(s)), Index[prime(s)]
end

function _map_from_gates_by_contraction(
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
    liouvillian_propagator_itensor(os, sites, dt; alg=Exact(), jump_ops=[], liouville_form=false)

Build the one-step Liouville propagator ``U = \\exp(dt \\, L)`` as a single
`ITensor` on `sites`.

When `liouville_form=false`, `os` is a physical Hamiltonian and `L` is built by
[`OpSum_Liouville`](@ref):

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
  ``\\exp(dt\\,L) \\approx \\prod_j \\exp(dt\\,L_j)`` using [`trotter_gates`](@ref),
  then contract the gate list with [`propagator_itensor_from_gates`](@ref).

Leg convention: unprimed `sites` are ket/output legs; `prime.(sites)` are
bra/input legs.

`materialize_method` is forwarded to `propagator_itensor_from_gates` when
`alg isa Trotter` (`:auto` selects `:basis` for one site, `:contract` otherwise).
"""
function liouvillian_propagator_itensor(
    os::OpSum,
    sites::AbstractVector{<:Index},
    dt::Real;
    alg=Exact(),
    jump_ops=Tuple{Number,String,Int}[],
    liouville_form::Bool=false,
    materialize_method::Symbol=:auto,
)
    L = liouville_form ? os : OpSum_Liouville(os, jump_ops)

    if alg isa Exact
        L_mpo = ITensorMPS.MPO(L, sites)
        L_dense = foldl(*, L_mpo)
        return itensor_exp(dt * L_dense)
    elseif alg isa Trotter
        gates = trotter_gates(L, sites, dt; alg=alg)

        method = materialize_method === :auto ?
            (length(sites) == 1 ? :basis : :contract) :
            materialize_method

        U, final_out = propagator_itensor_from_gates(gates, sites; method=method)

        for (old_out, s) in zip(final_out, sites)
            U = replaceind(U, old_out, prime(s))
        end

        return U
    end

    throw(ArgumentError("Unsupported exponentiation algorithm: $(typeof(alg))."))
end

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
[`trotter_gates`](@ref); `Trotter{2}()` is the default second-order choice.
The gates are applied `round(T/dt)` times with optional truncation (`maxdim`,
`cutoff`).

For exact single-step exponentiation on small Liouville spaces, use
[`liouvillian_propagator_itensor`](@ref) with `alg=Exact()` instead of TEBD.

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
    L = OpSum_Liouville(H, jump_ops)
    gates = trotter_gates(L, siteinds(state), dt; alg=alg)
    return _tebd_loop(state, gates, dt, T; maxdim, cutoff, verbose)
end
