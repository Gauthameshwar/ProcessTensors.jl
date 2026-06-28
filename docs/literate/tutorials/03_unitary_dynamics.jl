# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors #src
# SPDX-License-Identifier: MIT #src
# #src
# File: docs/literate/tutorials/03_unitary_dynamics.jl #src
# Contributor: Gauthameshwar S. #src
# #src
# Literate tutorial source: unitary TEBD/TDVP in Hilbert and Liouville space. #src

# # Unitary Dynamics
#
# This tutorial continues from [Liouville-Space Basics](@ref).
#
# So far we know how to build MPS/MPO objects and vectorize density matrices.
# Now we **evolve** them in time for a closed spin chain.
#
# The central question is: If we evolve a pure state in Hilbert space, and evolve 
# its density matrix in Liouville space, do we get the same observables?
#
# For unitary dynamics the answer should be yes. We will check this on a tiny
# four-site chain where exact diagonalization (ED) is still possible.

# ## Setup

using ITensors
import ITensorMPS
import LinearAlgebra
using ProcessTensors

#
# `ITensorMPS.jl` already provides MPS/MPO objects, `OpSum`, gate application,
# TEBD, and TDVP [ITensorMPS time evolution docs](https://docs.itensor.org/ITensorMPS/stable/tutorials/MPSTimeEvolution.html). 
# `ProcessTensors.jl` builds on this rather than replacing it.
#
# The nontrivial extension here is that the same time-evolution language is made
# available for:
#
# - Hilbert-space states, `MPS{Hilbert}`,
# - Hilbert-space operators, `MPO{Hilbert}`,
# - vectorized density matrices, `MPS{Liouville}`,
# - Liouville-space generators, `MPO{Liouville}`.
#
# This tutorial first reviews Hilbert-space TEBD/TDVP, then shows how the same
# physics can be evolved in Liouville space.

# ## Model and exact diagonalization
#
# We use a transverse-field Ising chain on $N=4$ spins:
#
# ```math
# H = H_X + H_{ZZ},
# \qquad
# H_{ZZ} = -J\sum_{j=1}^{N-1} S^z_j S^z_{j+1},
# \qquad
# H_X = -h\sum_{j=1}^{N} S^x_j.
# ```
#
# The initial state is the product state $|\uparrow\rangle^{\otimes N}$.
#
# Because $2^4 = 16$, we can form the dense Hamiltonian matrix and compare
# tensor-network evolution against the exact unitary
#
# ```math
# |\psi(t)\rangle = e^{-iHt}|\psi(0)\rangle.
# ```

N = 4
J = 1.0
h = 1.2
dt = 0.05
T = 1.0
maxdim = 32
cutoff = 1e-10
sample_times = collect(range(0, T; length=5))

function tfim_hamiltonian(N; J=1.0, h=1.2)
    os = OpSum()
    for j in 1:(N - 1)
        os += -J, "Sz", j, "Sz", j + 1
    end
    for j in 1:N
        os += -h, "Sx", j
    end
    return os
end

sites = siteinds("S=1/2", N)
H = tfim_hamiltonian(N; J, h)
H_mpo = MPO(H, sites)
ψ0 = MPS(sites, fill("Up", N))

println("Hamiltonian OpSum:")
println(H)

@assert ψ0 isa MPS{Hilbert}
@assert H_mpo isa MPO{Hilbert}

# !!! note "Small dense helper functions"
#     The next few functions contract MPOs to dense matrices for **validation
#     only** on this tiny chain. They are not part of the scalable
#     tensor-network workflow.

function dense_mpo_matrix(W, sites)
    T = W[1]
    for n in 2:length(W)
        T *= W[n]
    end
    D = prod(dim.(sites))
    A = Array(T, prime.(sites)..., sites...)
    return reshape(ComplexF64.(A), D, D)
end

function local_sz_matrices(sites, N)
    mats = Vector{Matrix{ComplexF64}}(undef, N)
    for j in 1:N
        os = OpSum()
        os += 1.0, "Sz", j
        mats[j] = dense_mpo_matrix(MPO(os, sites), sites)
    end
    return mats
end

function mean_sz_from_density(ρ_dense, Sz_dense, N)
    total = 0.0
    for j in 1:N
        total += real(LinearAlgebra.tr(Sz_dense[j] * ρ_dense))
    end
    return total / N
end

H_dense = dense_mpo_matrix(H_mpo, sites)
Sz_dense = local_sz_matrices(sites, N)

ψ0_dense = let
    Tψ = ψ0[1]
    for n in 2:N
        Tψ *= ψ0[n]
    end
    vec(ComplexF64.(Array(Tψ, sites...)))
end

# At time $t$ the exact energy and mean magnetization are
# $E(t)=\operatorname{Tr}(H\rho(t))$ and
# $\langle S^z\rangle = \frac{1}{N}\sum_j \operatorname{Tr}(S^z_j \rho(t))$,
# computed fully in the dense ED reference below.

function exact_energy_and_mz(t, H_dense, ψ0_dense, Sz_dense, N)
    ψt = iszero(t) ? ψ0_dense : LinearAlgebra.exp(-1im * t * H_dense) * ψ0_dense
    ρ = ψt * ψt'
    E = real(LinearAlgebra.tr(H_dense * ρ))
    mz = mean_sz_from_density(ρ, Sz_dense, N)
    return E, mz, ρ
end

E0, mz0, _ = exact_energy_and_mz(0.0, H_dense, ψ0_dense, Sz_dense, N)
println("Exact reference at t = 0:  E = ", E0, ",  mean ⟨Sz⟩ = ", mz0)

# ## TEBD time evolution
#
# **Time-evolving block decimation (TEBD)** approximates the short-time propagator
#
# ```math
# U(\Delta t) = e^{-iH\Delta t}
# ```
#
# by a product of **local gates**. The gates come from a Suzuki–Trotter
# factorization of $H = H_X + H_{ZZ}$.
#
# For a second-order Trotter step (`Trotter{2}()`), schematically
#
# ```math
# e^{-i(H_X + H_{ZZ})\Delta t} \approx
# e^{-iH_X \Delta t/2}\,
# e^{-iH_{ZZ}\Delta t}\,
# e^{-iH_X \Delta t/2}
# + \mathcal{O}(\Delta t^3).
# ```
#
# `ProcessTensors.jl` builds these gates through `trotter_gates` and applies them
# repeatedly in `tebd`.

# ### Inspecting the Trotter gates
#
# `trotter_gates` expands one Trotter step into local ITensor gates. ITensors
# currently supports this factorization for `Trotter{1}()` and `Trotter{2}()`.
# For the specified Hamiltonian, we would have four on-site terms and three 
# two-site terms corresponding to each term in the Hamiltonian. So in the 
# first-order Trotter, we would expect a total of seven gates, and for the 
# second-order Trotter, we would expect a twice of that.

println("Gates per Trotter step on this chain:")
for order in (1, 2)
    alg = Trotter{order}()
    step_gates = trotter_gates(H, sites, -im * dt; alg=alg)
    println("  Trotter{", order, "}: ", length(step_gates), " gates")
end

gates = trotter_gates(H, sites, -im * dt; alg=Trotter{2}())
println("Indices of the first Trotter{2} gate: ", inds(gates[1]))

# !!! note "Higher Trotter orders"
#     Types such as `Trotter{4}()` exist in ITensors, but `trotter_gates` is
#     currently implemented only for orders `1` and `2`. In this tutorial we
#     compare those two orders in the TEBD accuracy check below.

# ### Evolving with `tebd`
#
# The call
#
# ```julia
# tebd(ψ, H, dt, Δt; alg=Trotter{2}())
# ```
#
# applies `round(Δt/dt)` Trotter steps to evolve from the current MPS for a
# duration `Δt`. Truncation is controlled by `maxdim` and `cutoff`.
#
# The energy is $\langle H\rangle = \langle\psi|H|\psi\rangle$, computed as
# `real(inner(ψ', H_mpo, ψ))`.

ψ1 = tebd(ψ0, H, dt, dt; alg=Trotter{2}(), maxdim=maxdim, cutoff=cutoff)

E1 = real(inner(ψ1', H_mpo, ψ1))
mz1 = sum(expect(ψ1, "Sz")) / N

println("After one TEBD step (Δt = ", dt, "):")
println("  energy    = ", E1)
println("  mean ⟨Sz⟩ = ", mz1)

@assert ψ1 isa MPS{Hilbert}

# ### Trotter order comparison
#
# Higher Trotter order usually reduces splitting error at fixed `dt`. Here is a
# compact check at the final time $T$:

_, _, ρ_exact_T = exact_energy_and_mz(T, H_dense, ψ0_dense, Sz_dense, N)

println()
println("TEBD density-matrix error at t = ", T, ":")
for alg in (Trotter{1}(), Trotter{2}())
    ψ_alg = tebd(ψ0, H, dt, T; alg, maxdim=maxdim, cutoff=cutoff)
    ρ_alg = dense_mpo_matrix(to_dm(ψ_alg), sites)
    err = LinearAlgebra.norm(ρ_alg - ρ_exact_T) / max(LinearAlgebra.norm(ρ_exact_T), eps())
    println("  ", typeof(alg), "  ρ_err = ", round(err, digits=4))
end

# We must note, however, that this accuracy comes at a cost of more gate operations 
# while contracting them with the state. In principle, we could go to even higher-order 
# Trotter, but the numerical cost would be significant. 
# 
# ### TEBD vs exact evolution
#
# Instead of looking only at the final time, let us sample the trajectory. This
# shows whether the tensor-network evolution follows the exact curve, not just
# whether it lands close at one point.

function compare_tebd(sample_times, ψ0, H, H_mpo, sites, H_dense, ψ0_dense, Sz_dense)
    ψ = ψ0
    t_prev = 0.0
    println()
    println("TEBD vs exact (Trotter{2}):")
    println("  t      E_exact    E_tebd     mz_exact   mz_tebd")
    println("  " * "-"^52)
    for t in sample_times
        if t > 0.0
            ψ = tebd(ψ, H, dt, t - t_prev; alg=Trotter{2}(), maxdim=maxdim, cutoff=cutoff)
            t_prev = t
        end
        E_ex, mz_ex, ρ_ex = exact_energy_and_mz(t, H_dense, ψ0_dense, Sz_dense, N)
        E_tebd = real(inner(ψ', H_mpo, ψ))
        mz_tebd = sum(expect(ψ, "Sz")) / N
        ρ_tebd = dense_mpo_matrix(to_dm(ψ), sites)
        ρ_err = LinearAlgebra.norm(ρ_tebd - ρ_ex) / max(LinearAlgebra.norm(ρ_ex), eps())
        println("  $(lpad(round(t, digits=2), 5))  ",
                lpad(round(E_ex, digits=4), 9), "  ",
                lpad(round(E_tebd, digits=4), 9), "  ",
                lpad(round(mz_ex, digits=4), 9), "  ",
                lpad(round(mz_tebd, digits=4), 9),
                "   ρ_err=", round(ρ_err, digits=4))
    end
end

compare_tebd(sample_times, ψ0, H, H_mpo, sites, H_dense, ψ0_dense, Sz_dense)

# !!! note "Energy conservation and numerical drift"
#     In exact unitary dynamics with a time-independent Hamiltonian, energy is
#     conserved. Any drift in the TEBD/TDVP energy is a numerical error from
#     Trotter splitting, time-step error, or MPS truncation.

# ## TDVP time evolution
#
# **Time-dependent variational principle (TDVP)** takes a different view. Instead
# of applying a product of fixed gates, TDVP projects the Schrödinger equation
#
# ```math
# \frac{d}{dt}|\psi\rangle = -iH|\psi\rangle
# ```
#
# onto the tangent space of the MPS manifold at the current state.
#
# `ITensorMPS.jl` implements the algorithm. `ProcessTensors.jl` forwards the
# call on the wrapped `.core` object and returns `MPS{Hilbert}`.
#
# The Hamiltonian is passed as an `MPO{Hilbert}` and the evolution time enters
# as a **complex** number:
#
# ```math
# |\psi(t)\rangle \approx \mathrm{TDVP}\big(H,\,-it,\,|\psi(0)\rangle\big).
# ```
#
# So the second argument is `-im * t`, and the integrator step is `-im * dt`.

# ### Evolving with `tdvp`

ψ_tdvp = tdvp(
    H_mpo,
    -im * dt,
    ψ0;
    time_step=-im * dt,
    nsite=2,
    maxdim=maxdim,
    cutoff=cutoff,
    outputlevel=0,
)

E_tdvp = real(inner(ψ_tdvp', H_mpo, ψ_tdvp))
mz_tdvp = sum(expect(ψ_tdvp, "Sz")) / N

println("After one TDVP step (Δt = ", dt, "):")
println("  energy    = ", E_tdvp)
println("  mean ⟨Sz⟩ = ", mz_tdvp)

@assert ψ_tdvp isa MPS{Hilbert}

# ### TDVP vs exact evolution

function compare_tdvp(sample_times, ψ0, H_mpo, sites, H_dense, ψ0_dense, Sz_dense)
    ψ = ψ0
    t_prev = 0.0
    println()
    println("TDVP vs exact:")
    println("  t      E_exact    E_tdvp     mz_exact   mz_tdvp")
    println("  " * "-"^52)
    for t in sample_times
        if t > 0.0
            ψ = tdvp(
                H_mpo,
                -im * (t - t_prev),
                ψ;
                time_step=-im * dt,
                nsite=2,
                maxdim=maxdim,
                cutoff=cutoff,
                outputlevel=0,
            )
            t_prev = t
        end
        E_ex, mz_ex, ρ_ex = exact_energy_and_mz(t, H_dense, ψ0_dense, Sz_dense, N)
        E_tdvp = real(inner(ψ', H_mpo, ψ))
        mz_tdvp = sum(expect(ψ, "Sz")) / N
        ρ_tdvp = dense_mpo_matrix(to_dm(ψ), sites)
        ρ_err = LinearAlgebra.norm(ρ_tdvp - ρ_ex) / max(LinearAlgebra.norm(ρ_ex), eps())
        println("  $(lpad(round(t, digits=2), 5))  ",
                lpad(round(E_ex, digits=4), 9), "  ",
                lpad(round(E_tdvp, digits=4), 9), "  ",
                lpad(round(mz_ex, digits=4), 9), "  ",
                lpad(round(mz_tdvp, digits=4), 9),
                "   ρ_err=", round(ρ_err, digits=4))
    end
end

compare_tdvp(sample_times, ψ0, H_mpo, sites, H_dense, ψ0_dense, Sz_dense)

# ## Hilbert versus Liouville evolution
#
# The same unitary physics can be written in Liouville space. A pure state
# $\rho = |\psi\rangle\langle\psi|$ obeys
#
# ```math
# \frac{d\rho}{dt} = -i[H,\rho],
# \qquad
# |\rho(t)\rangle\rangle = e^{\mathcal{L}_H t}|\rho(0)\rangle\rangle,
# ```
#
# with $\mathcal{L}_H = -iH_L + iH_R$. The Liouville generator is available as
# `MPO_Liouville(H, sites_L)`.
#
# For TDVP the time argument is **`T`**, not `-im * T`, because the factor
# $-i$ is already inside the Liouville MPO.

ρ0 = to_dm(ψ0)
sites_L = liouv_sites(sites)
ρL0 = to_liouville(ρ0; sites=sites_L)
L_mpo = MPO_Liouville(H, sites_L)

@assert ρL0 isa MPS{Liouville}

# ### Evolving with `tdvp` in Liouville space
#
# The Liouville analogue of the Hilbert TDVP call is
#
# ```julia
# tdvp(L_mpo, Δt, ρL0; time_step=dt, nsite=2, maxdim, cutoff)
# ```
#
# Here `ρL0` is an `MPS{Liouville}` and `L_mpo` is the Liouville generator from
# `MPO_Liouville`. The time argument is **`Δt`**, not `-im * Δt`.

ρL1 = tdvp(
    L_mpo,
    dt,
    ρL0;
    time_step=dt,
    nsite=2,
    maxdim=maxdim,
    cutoff=cutoff,
    outputlevel=0,
)

ρ1_liouville = dense_mpo_matrix(to_hilbert(ρL1), sites)
E_l1 = real(LinearAlgebra.tr(H_dense * ρ1_liouville))
mz_l1 = mean_sz_from_density(ρ1_liouville, Sz_dense, N)

println("After one Liouville TDVP step (Δt = ", dt, "):")
println("  energy    = ", E_l1)
println("  mean ⟨Sz⟩ = ", mz_l1)
println("  Tr(ρ)     = ", LinearAlgebra.tr(ρ1_liouville))

@assert ρL1 isa MPS{Liouville}

# ### Hilbert vs Liouville TDVP
#
# Here we perform the time evolution of our initial state in a TFIM in both the 
# Hilbert and Liouville spaces and see if they match.
# The strongest check is whether the density matrices agree, not just individual
# observables. We also print $\operatorname{Tr}(\rho)$ from the Liouville route.

function compare_hilbert_liouville(sample_times, ψ0, H_mpo, L_mpo, sites, H_dense, Sz_dense)
    ψ = ψ0
    ρL = ρL0
    t_prev = 0.0
    println()
    println("Hilbert vs Liouville TDVP:")
    println("  t      E_Hilbert  E_Liouv    mz_Hilbert mz_Liouv   Tr(ρ_L)   ρ_err")
    println("  " * "-"^68)
    for t in sample_times
        if t > 0.0
            Δt = t - t_prev
            ψ = tdvp(
                H_mpo,
                -im * Δt,
                ψ;
                time_step=-im * dt,
                nsite=2,
                maxdim=maxdim,
                cutoff=cutoff,
                outputlevel=0,
            )
            ρL = tdvp(
                L_mpo,
                Δt,
                ρL;
                time_step=dt,
                nsite=2,
                maxdim=maxdim,
                cutoff=cutoff,
                outputlevel=0,
            )
            t_prev = t
        end
        E_h = real(inner(ψ', H_mpo, ψ))
        mz_h = sum(expect(ψ, "Sz")) / N
        ρ_h = dense_mpo_matrix(to_dm(ψ), sites)
        ρ_l = dense_mpo_matrix(to_hilbert(ρL), sites)
        E_l = real(LinearAlgebra.tr(H_dense * ρ_l))
        mz_l = mean_sz_from_density(ρ_l, Sz_dense, N)
        tr_ρ_l = LinearAlgebra.tr(ρ_l)
        ρ_hl_err = LinearAlgebra.norm(ρ_h - ρ_l) / max(LinearAlgebra.norm(ρ_h), eps())
        println("  $(lpad(round(t, digits=2), 5))  ",
                lpad(round(E_h, digits=4), 9), "  ",
                lpad(round(E_l, digits=4), 9), "  ",
                lpad(round(mz_h, digits=4), 9), "  ",
                lpad(round(mz_l, digits=4), 9), "  ",
                lpad(round(real(tr_ρ_l), digits=4), 7), "  ",
                round(ρ_hl_err, digits=4))
    end
end

compare_hilbert_liouville(sample_times, ψ0, H_mpo, L_mpo, sites, H_dense, Sz_dense)

# ### Summary
#
# - `ITensorMPS.jl` provides TEBD/TDVP; `ProcessTensors.jl` extends them to typed
#   Hilbert/Liouville MPS and MPO objects.
# - TEBD approximates $e^{-iH\Delta t}$ by Trotter gates from `OpSum`.
# - TDVP projects Schrödinger evolution onto the MPS manifold; pass `-im * t`.
# - Liouville TDVP evolves `MPS{Liouville}` with `MPO_Liouville`; pass `T`.
# - Reuse `sites_L` across `to_liouville` and `MPO_Liouville`.
#
# Next: [Dissipative Dynamics](@ref), where jump terms make Liouville space essential.
