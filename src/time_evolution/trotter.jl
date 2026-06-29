# src/time_evolution/trotter.jl

import ITensors: ITensor, Index, exp
import ITensors.Ops: Trotter, Prod

trotter_order(::Trotter{Order}) where {Order} = Order

function _trotter_gates_itensors(
    os::OpSum,
    sites::AbstractVector{<:Index},
    τ::Number,
    order::Integer,
)
    order in (1, 2) ||
        throw(ArgumentError("_trotter_gates_itensors: expected order 1 or 2, got $order."))
    lazy = exp(τ * os; alg=Trotter{order}())
    prod = Prod{ITensor}(lazy, collect(sites))
    return collect(ITensor, only(prod.args))
end

function _yoshida_fractal_weights(order::Integer)
    iseven(order) && order >= 4 ||
        throw(ArgumentError("_yoshida_fractal_weights: expected even order >= 4, got $order."))
    u1 = 1 / (2 - 2^(1 / (order - 1)))
    u0 = 1 - 2u1
    return u1, u0
end

function _trotter_gates_one_step(
    os::OpSum,
    sites::AbstractVector{<:Index},
    τ::Number,
    order::Integer,
)
    if order == 1 || order == 2
        return _trotter_gates_itensors(os, sites, τ, order)
    elseif iseven(order)
        lower_order = order - 2
        u1, u0 = _yoshida_fractal_weights(order)
        gates = ITensor[]
        for weight in (u1, u0, u1)
            append!(gates, _trotter_gates_one_step(os, sites, weight * τ, lower_order))
        end
        return gates
    end

    throw(
        ArgumentError(
            "Unsupported Trotter order $order. Supported orders are 1, 2, and even orders >= 4.",
        ),
    )
end

function _repeat_trotter_gates(gates::Vector{ITensor}, nsteps::Integer)
    nsteps == 1 && return gates
    repeated = ITensor[]
    for _ in 1:nsteps
        append!(repeated, gates)
    end
    return repeated
end

"""
    trotter_gates(os, sites, τ; alg=Trotter{2}())

Factorize ``\\exp(\\tau \\, \\mathrm{os})`` into a vector of ITensor gates using a
Suzuki–Trotter decomposition.

Schematically, for `alg = Trotter{2}()` the decomposition has the form

```math
\\exp(\\tau \\, \\mathrm{os}) \\approx
\\prod_{\\text{groups}} \\exp\\!\\left(\\frac{\\tau}{2}\\, \\mathrm{os}_{\\text{group}}\\right)
\\cdots
\\prod_{\\text{groups}} \\exp\\!\\left(\\frac{\\tau}{2}\\, \\mathrm{os}_{\\text{group}}\\right),
```

Orders `1` and `2` use the `ITensors.Ops` factorization. Even orders `n >= 4`
are built recursively with Yoshida's symmetric fractal composition, extending
the upstream ITensors support beyond `Trotter{1}()` and `Trotter{2}()`.

The `nsteps` field of `alg` matches `ITensors.Ops`: the generator is scaled by
`1/nsteps` and the resulting gate list is repeated `nsteps` times.

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
    order = trotter_order(alg)
    sub_τ = τ / alg.nsteps
    gates = _trotter_gates_one_step(os, sites, sub_τ, order)
    return _repeat_trotter_gates(gates, alg.nsteps)
end
