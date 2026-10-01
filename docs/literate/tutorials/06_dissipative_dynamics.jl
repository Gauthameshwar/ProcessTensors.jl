# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors #src
# SPDX-License-Identifier: MIT #src
# #src
# File: docs/literate/tutorials/06_dissipative_dynamics.jl #src
# Contributor: Gauthameshwar S. #src
# #src
# Literate tutorial source: dissipative TEBD/TDVP and concise physicality checks. #src

# # Dissipative Dynamics
#
# This tutorial evolves a small spin system with coherent interactions and local
# amplitude damping. We construct its Liouville-space state, evolve it with TEBD
# and TDVP, and check both the physical output and numerical accuracy.
#
# The model is a Markovian master equation with specified jump operators.
# These time-evolution tools complement the process-tensor workflows introduced
# in [Construct a process tensor](@ref). For the algorithm introductions,
# see [Unitary Dynamics](@ref); for vectorisation, see [Liouville-Space Basics](@ref).

# ## Setup

using ITensors
import ITensorMPS
import LinearAlgebra as LA
using ProcessTensors
using ITensors.Ops: Trotter

# Two helpers are reused for both algorithms. Dense reconstruction is suitable
# only for this two-spin demonstration. Diagnostics are returned as named tuples,
# so Julia displays their labels without a custom printing routine.

function dense_density(ρL, sites)
    W = to_hilbert(ρL)
    D = prod(dim.(sites))
    return reshape(ComplexF64.(Array(foldl(*, W), prime.(sites)..., sites...)), D, D)
end

function density_checks(ρ)
    scale = max(LA.norm(ρ), eps(Float64))
    ρh = LA.Hermitian((ρ + ρ') / 2)
    return (; trace_error=abs(LA.tr(ρ) - 1),
            hermiticity_error=LA.norm(ρ - ρ') / scale,
            min_eigenvalue=minimum(LA.eigvals(ρh)))
end;

# ## [A spin pair with amplitude damping](@id dissipative-lindblad-mpo)
#
# We use spin operators $S^\alpha=\sigma^\alpha/2$ and set $\hbar=1$:
#
# ```math
# H=J S^z_1S^z_2+h(S^x_1+S^x_2),
# \qquad
# \dot\rho=-i[H,\rho]+\gamma\sum_{j=1}^2\mathcal D[S^-_j](\rho),
# ```
#
# where $\mathcal D[L](\rho)=L\rho L^\dagger-\{L^\dagger L,\rho\}/2$.
# Damping transfers population from `Up` to `Dn`; the transverse field mixes
# those states, and the interaction can generate correlations. We start with
# both spins up, so the initial mean magnetisation is $1/2$.
#
# A single two-spin model is enough to demonstrate both algorithms and retains
# a small dense reference. All Liouville objects share the same `sites_L`.

J, h, γ = 1.0, 0.4, 0.15
dt, T = 0.025, 0.5
maxdim, cutoff = 16, 1e-12
sites = siteinds("S=1/2", 2)
sites_L = liouv_sites(sites)
ψ0 = MPS(sites, ["Up", "Up"])
ρL0 = to_liouville(to_dm(ψ0); sites=sites_L)

H = OpSum()
H += J, "Sz", 1, "Sz", 2
H += h, "Sx", 1
H += h, "Sx", 2
jump_ops = [(γ, "S-", 1), (γ, "S-", 2)]
L_mpo = liouvillian_mpo(H, sites_L; jump_ops=jump_ops);

# !!! note "Jump rates and generator conventions"
#     `(γ, "S-", j)` contributes `γ * D[S-]` at site `j`; the first entry is
#     the rate, not its square root. `L_mpo` already includes the commutator's
#     factor $-i$ and acts as $\partial_t|\rho\rangle\rangle=\mathcal L|\rho\rangle\rangle$.
#     The propagator is $e^{T\mathcal L}$.

# ## [Liouville-space TEBD](@id dissipative-evolving-density-matrix)
#
# TEBD approximates $e^{\mathcal L\Delta t}$ with local gates and truncates
# the resulting MPS bonds. Supply the Hamiltonian `OpSum` and `jump_ops`;
# the Liouville generator is assembled internally from the state's indices.

ρL_tebd = tebd(
    ρL0, H, dt, T;
    jump_ops=jump_ops, alg=Trotter{2}(),
    maxdim=maxdim, cutoff=cutoff, progress=false,
);

# To read out the magnetisation, vectorise the observable on the same indices.
# The overlap gives $\operatorname{Tr}(\bar S^z\rho)$, with
# $\bar S^z=(S^z_1+S^z_2)/2$. We retain the complex result so any spurious
# imaginary part remains visible.

Sz_mean = OpSum()
Sz_mean += 0.5, "Sz", 1
Sz_mean += 0.5, "Sz", 2
Sz_L = to_liouville(MPO(Sz_mean, sites); sites=sites_L);
println((initial=inner(Sz_L, ρL0), tebd=inner(Sz_L, ρL_tebd)))

# !!! note "Dissipative splitting and physicality"
#     A Trotter approximation is CPTP if each factor is itself a forward-time
#     CPTP map. Splitting a dissipator into individual algebraic terms does not
#     automatically meet that condition. Higher-order compositions can also
#     contain negative substeps. Use convergence and physicality checks rather
#     than assuming that a higher Trotter order guarantees a physical result.
#     Bond truncation adds a separate approximation.

# ## Liouville-space TDVP
#
# TDVP evolves the vectorised density within an MPS manifold, using `L_mpo`
# directly. We use two-site updates so the initial product density can develop
# operator-space correlations. Its total time and step are real because the
# generator already contains the Hamiltonian factor $-i$.

ρL_tdvp = tdvp(
    L_mpo, T, ρL0;
    time_step=dt, nsite=2, maxdim=maxdim, cutoff=cutoff,
    normalize=false, updater_kwargs=(; ishermitian=false), outputlevel=0,
);

#-
println((tebd=inner(Sz_L, ρL_tebd), tdvp=inner(Sz_L, ρL_tdvp)))

# !!! info "What changes from pure-state TDVP?"
#     The Liouvillian is generally non-Hermitian, and the Euclidean norm of a
#     Liouville vector measures purity, not trace. We therefore use a
#     non-Hermitian local solver and keep `normalize=false`. Physical energy
#     need not be conserved in this dissipative model. One-site TDVP still has
#     fixed bond dimensions; GSE can enrich its basis but does not enforce
#     trace or positivity. See [Unitary Dynamics](@ref) for that distinction.

# ## [Checking physicality](@id dynamics-physicality)
#
# A physical deterministic output has unit trace, is Hermitian, and is positive
# semidefinite. TEBD truncation and splitting, or TDVP projection and numerical
# integration, do not generally impose all of these constraints on a Liouville
# MPS. This applies even when the underlying evolution is unitary.
#
# Reconstruct the small density matrices and inspect three labelled numbers:

ρ_tebd = dense_density(ρL_tebd, sites)
ρ_tdvp = dense_density(ρL_tdvp, sites)
tebd_checks = density_checks(ρ_tebd)
tdvp_checks = density_checks(ρ_tdvp)
println(tebd_checks)
println(tdvp_checks)

# | Diagnostic | Interpretation |
# |:--|:--|
# | `trace_error` | $\vert\operatorname{Tr}\rho-1\vert$, including any imaginary trace error |
# | `hermiticity_error` | Relative Frobenius norm of $\rho-\rho^\dagger$ |
# | `min_eigenvalue` | Smallest eigenvalue of $(\rho+\rho^\dagger)/2$ |
#
# Interpret the eigenvalue together with the Hermiticity error: a non-Hermitian
# matrix is already unphysical. The Hermitian part is formed only to diagnose
# the result; the evolved state is not replaced by it. Tiny negative values can
# reflect numerical error and should shrink under suitable convergence checks.

@assert tebd_checks.trace_error < 1e-3
@assert tdvp_checks.trace_error < 1e-3
@assert tebd_checks.hermiticity_error < 1e-3
@assert tdvp_checks.hermiticity_error < 1e-3
@assert tebd_checks.min_eigenvalue > -1e-3
@assert tdvp_checks.min_eigenvalue > -1e-3

# !!! warning "Normalisation does not repair positivity"
#     Rescaling by the trace can restore unit trace, but it does not remove
#     negative eigenvalues or projection errors. Euclidean normalisation is
#     different again: it fixes $\operatorname{Tr}(\rho^\dagger\rho)$.
#     For a large network, trace and observable checks remain accessible;
#     positivity checks on small reduced states are useful but do not certify
#     positivity of the complete many-body density operator.

# ### Accuracy against a small dense reference
#
# Passing physicality checks does not establish accuracy. Here the Liouville
# dimension is only $4^2=16$, so we can compare both final states with the dense
# matrix exponential. Extract the generator and state using the same local
# Liouville ordering; no global density-matrix reshuffling is needed.

D_L = prod(dim.(sites_L))
L_dense = reshape(
    ComplexF64.(Array(foldl(*, L_mpo), prime.(sites_L)..., sites_L...)), D_L, D_L,
)
v0 = vec(ComplexF64.(Array(foldl(*, ρL0), sites_L...)))
v_exact = LA.exp(T * L_dense) * v0
v_tebd = vec(ComplexF64.(Array(foldl(*, ρL_tebd), sites_L...)))
v_tdvp = vec(ComplexF64.(Array(foldl(*, ρL_tdvp), sites_L...)))

errors = (
    tebd=LA.norm(v_tebd - v_exact) / LA.norm(v_exact),
    tdvp=LA.norm(v_tdvp - v_exact) / LA.norm(v_exact),
)
println(errors)
@assert all(isfinite, values(errors))
@assert maximum(values(errors)) < 1e-3

# These assertions are regression checks for this tiny model, not general
# physicality or accuracy thresholds. Tighten `dt` and the bond controls to
# establish convergence for the observables in a larger calculation. Monitor
# intermediate times as well as the endpoint when studying a full trajectory.
#
# The workflow is now complete: specify `H` and the jumps, construct the
# Liouville state, evolve, then inspect observables and diagnostics. The examples
# below apply this pattern to larger physical models.
#
# !!! related "Continue learning"
#     - [Dissipative spin chain](../examples/dissipative_spin.md): interacting spins with local damping.
#     - [Driven-dissipative Bose–Hubbard](../examples/driven_dissipative_bose_hubbard.md): driven bosons with loss.
#     - [Quantum States and Liouville Space](../theory/liouville_space.md): vectorisation and map conventions.
#     - [Construct a process tensor](@ref): retain environmental influence across multiple intervention times.
