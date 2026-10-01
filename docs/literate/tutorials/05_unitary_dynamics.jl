# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors #src
# SPDX-License-Identifier: MIT #src
# #src
# File: docs/literate/tutorials/05_unitary_dynamics.jl #src
# Contributor: Gauthameshwar S. #src
# #src
# Literate tutorial source: a short introduction to unitary TEBD and TDVP. #src

# # Unitary Dynamics
#
# This tutorial introduces two ways to evolve a closed spin chain: time-evolving
# block decimation (TEBD) and the time-dependent variational principle (TDVP).
# We build one small model, evolve it with each method, and interpret a compact
# set of diagnostics. A final comparison shows how the same physics appears in
# Liouville space.
#
# These are additional time-evolution tools provided by `ProcessTensors.jl`.
# For construction and reuse of a process tensor, start with
# [Construct a process tensor](@ref).

# ## Setup
#
# `ProcessTensors.jl` supplies the typed Hilbert/Liouville interface and the
# `tebd` driver. Its `tdvp` methods forward to `ITensorMPS.jl`.

using ITensors
import ITensorMPS
import LinearAlgebra as LA
using ProcessTensors
using ITensors.Ops: Trotter

# Two short helpers keep repeated diagnostics readable. Dense conversion is
# used only for this four-spin check; its cost grows exponentially with size.
# `unitary_checks` returns labelled values using Julia's built-in named tuples.

function dense_matrix(W, sites)
    D = prod(dim.(sites))
    return reshape(ComplexF64.(Array(foldl(*, W), prime.(sites)..., sites...)), D, D)
end

function unitary_checks(ψ, H_mpo, initial_energy)
    norm2 = real(inner(ψ, ψ))
    energy = real(inner(ψ', H_mpo, ψ)) / norm2
    return (; norm_error=abs(norm2 - 1), energy_drift=abs(energy - initial_energy),
            mean_sz=sum(expect(ψ, "Sz")) / length(ψ), bond=maxlinkdim(ψ))
end;

# ## A small Ising chain
#
# We use open boundaries and spin operators $S^\alpha=\sigma^\alpha/2$:
#
# ```math
# H=-J\sum_{j=1}^{N-1}S^z_jS^z_{j+1}-h\sum_{j=1}^{N}S^x_j,
# \qquad |\psi_0\rangle=|\uparrow\rangle^{\otimes N}.
# ```
#
# The transverse field rotates the initially polarised spins; the interaction
# can generate entanglement. The mean longitudinal magnetisation changes while
# the energy of this time-independent Hamiltonian remains constant in exact
# evolution. We use $\hbar=1$.

function ising_hamiltonian(N, J, h)
    os = OpSum()
    for j in 1:(N - 1)
        os += -J, "Sz", j, "Sz", j + 1
    end
    for j in 1:N
        os += -h, "Sx", j
    end
    return os
end

N = 4
J, h = 1.0, 1.2
dt, T = 0.05, 1.0
maxdim, cutoff = 32, 1e-10
sites = siteinds("S=1/2", N)
ψ0 = MPS(sites, fill("Up", N))
H = ising_hamiltonian(N, J, h)
H_mpo = MPO(H, sites)
E0 = real(inner(ψ0', H_mpo, ψ0));

#-
println(unitary_checks(ψ0, H_mpo, E0))
@assert isapprox(real(inner(ψ0, ψ0)), 1; atol=1e-12)

# ## TEBD: apply local gates
#
# TEBD splits a short-time propagator into local gates, applies them to the
# MPS, and truncates the resulting bonds. For two Hamiltonian pieces, a
# second-order splitting has the schematic form
#
# ```math
# e^{-i(H_A+H_B)\Delta t}
# =e^{-iH_A\Delta t/2}e^{-iH_B\Delta t}e^{-iH_A\Delta t/2}
# +\mathcal O(\Delta t^3).
# ```
#
# The package constructs gates from the terms of the Hamiltonian `OpSum`.
# Pass real durations `dt` and `T`; for a Hilbert MPS, `tebd` supplies the
# factor $-i$. Choose `T/dt` to be an integer.

ψ_tebd = tebd(
    ψ0, H, dt, T;
    alg=Trotter{2}(), maxdim=maxdim, cutoff=cutoff, progress=false,
);

#-
println(unitary_checks(ψ_tebd, H_mpo, E0))

# !!! note "Two separate accuracy controls"
#     Smaller `dt` reduces Trotter splitting error. Larger `maxdim` and a tighter
#     `cutoff` reduce bond truncation. Second-order splitting has a global error
#     of order $\Delta t^2$ at fixed duration before truncation dominates.
#     Higher orders require more gates and do not remove truncation error.
#     Check convergence in both the time step and the retained bond space.

# ## TDVP: evolve within an MPS manifold
#
# TDVP projects $\partial_t|\psi\rangle=-iH|\psi\rangle$ onto the tangent
# space of an MPS manifold and integrates the resulting equations. It works
# with the Hamiltonian MPO, so it is also useful when the Hamiltonian is less
# convenient to express as a short sequence of local gates.
#
# Here `nsite=2` updates two neighbouring tensors at a time, allowing bond
# growth followed by truncation. The MPO is $H$, so both the total evolution
# parameter and the step include $-i$.

ψ_tdvp = tdvp(
    H_mpo, -im * T, ψ0;
    time_step=-im * dt, nsite=2, maxdim=maxdim, cutoff=cutoff,
    normalize=false, outputlevel=0,
);

#-
println(unitary_checks(ψ_tdvp, H_mpo, E0))

# !!! info "One-site TDVP and global subspace expansion"
#     One-site TDVP (`nsite=1`) keeps the existing bond dimensions. Starting from
#     a product state, increasing `maxdim` alone does not supply the missing
#     entangled directions. Global subspace expansion (GSE) enriches the bond
#     basis with Krylov directions before one-site evolution. For this short
#     tutorial, two-site TDVP provides bond growth directly.
#
# !!! note "Conservation is useful, but not an accuracy certificate"
#     Ideal fixed-manifold Hilbert-space TDVP preserves norm and energy for a
#     time-independent Hermitian Hamiltonian. Finite solver tolerances and
#     two-site truncation can introduce drift. A restricted one-site trajectory
#     can conserve energy while giving inaccurate observables. TDVP also has
#     projection and integration errors; it is not automatically more accurate
#     than TEBD.

# ## A compact check against exact evolution
#
# Four spins give a dense Hamiltonian of size $16\times16$. We use its matrix
# exponential once to check the final density operator. This comparison is
# insensitive to an overall phase of the wavefunction and tests more than one
# observable. It is a small-system reference, not a scalable evolution method.

H_dense = dense_matrix(H_mpo, sites)
ψ0_dense = vec(ComplexF64.(Array(foldl(*, ψ0), sites...)))
ψ_exact = LA.exp(-im * T * H_dense) * ψ0_dense
ρ_exact = ψ_exact * ψ_exact'
ρ_tebd = dense_matrix(to_dm(ψ_tebd), sites)
ρ_tdvp = dense_matrix(to_dm(ψ_tdvp), sites)

errors = (
    tebd=LA.norm(ρ_tebd - ρ_exact) / LA.norm(ρ_exact),
    tdvp=LA.norm(ρ_tdvp - ρ_exact) / LA.norm(ρ_exact),
)
println(errors)
@assert all(isfinite, values(errors))
@assert maximum(values(errors)) < 1e-2

# The displayed values are relative Frobenius errors. The assertions are loose
# regression checks for this small demonstration, not general accuracy targets.
# The norm and energy diagnostics above provide complementary information.
# For a convergence study, repeat with `dt/2` and then tighter bond controls;
# agreeing with another approximate method alone is not a reference solution.

# ## [Hilbert versus Liouville evolution](@id hilbert-liouville-tdvp)
#
# The same closed-system density operator obeys
#
# ```math
# \partial_t|\rho\rangle\rangle=\mathcal L_H|\rho\rangle\rangle,
# \qquad \mathcal L_H\rho=-i[H,\rho].
# ```
#
# Reuse the same Liouville indices for the initial density and generator.
# Both TEBD and TDVP can evolve this representation.

sites_L = liouv_sites(sites)
ρL0 = to_liouville(to_dm(ψ0); sites=sites_L)
L_mpo = liouvillian_mpo(H, sites_L);

#-
ρL_tebd = tebd(
    ρL0, H, dt, T;
    alg=Trotter{2}(), maxdim=maxdim, cutoff=cutoff, progress=false,
);
ρL_tdvp = tdvp(
    L_mpo, T, ρL0;
    time_step=dt, nsite=2, maxdim=maxdim, cutoff=cutoff,
    normalize=false, updater_kwargs=(; ishermitian=false), outputlevel=0,
);

# !!! note "The generator determines the TDVP time argument"
#     With `H_mpo`, pass `-im * T` and `time_step=-im * dt`.
#     With `L_mpo`, pass `T` and `time_step=dt`: the generator already contains
#     $-i$. A Liouvillian is generally non-Hermitian, so the local solver is
#     told this explicitly. `normalize=false` avoids normalising the Liouville
#     vector's Euclidean norm, which represents purity rather than trace.

ρ_from_L_tebd = dense_matrix(to_hilbert(ρL_tebd), sites)
ρ_from_L_tdvp = dense_matrix(to_hilbert(ρL_tdvp), sites)
liouville_errors = (
    tebd=LA.norm(ρ_from_L_tebd - ρ_exact) / LA.norm(ρ_exact),
    tdvp=LA.norm(ρ_from_L_tdvp - ρ_exact) / LA.norm(ρ_exact),
)
println(liouville_errors)
@assert all(isfinite, values(liouville_errors))
@assert maximum(values(liouville_errors)) < 1e-2

# !!! info "Different representations, different numerical constraints"
#     An approximate pure-state MPS still defines a positive operator
#     $|\psi\rangle\langle\psi|$, although its norm and observables can have
#     errors. A general Liouville MPS does not enforce trace, Hermiticity, or
#     positivity. Hilbert-space TDVP conservation arguments do not automatically
#     protect the physical energy $\operatorname{Tr}(H\rho)$ in Liouville space.
#     See [Checking physicality](@ref dynamics-physicality) for practical checks.
#
# Liouville evolution also has a larger local dimension. For a pure state with
# Schmidt rank $\chi$, its density operator has operator-Schmidt rank $\chi^2$
# across the same cut. The exact physics agrees, but the two numerical
# representations can require different bond dimensions and show different errors.
#
# !!! related "Continue learning"
#     - [Dissipative Dynamics](@ref): add jump operators and check density-matrix physicality.
#     - [Laser-driven TDVP dynamics](../examples/laser_driven_tdvp.md): time-dependent spin driving.
#     - [Construct a process tensor](@ref): build a reusable multi-time process.
