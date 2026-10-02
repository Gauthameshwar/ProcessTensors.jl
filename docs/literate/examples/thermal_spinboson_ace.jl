# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors #src
# SPDX-License-Identifier: MIT #src
# #src
# File: docs/literate/examples/thermal_spinboson_ace.jl #src
# Contributor: Gauthameshwar S. #src
# #src
# Literate example: driven population oscillations in a thermal bosonic bath. #src

# # Thermal spin-boson dynamics using ACE
#
# How does a thermal bath change the Rabi oscillations of a driven two-level
# system? We sample an Ohmic spectral density into oscillator modes, prepare
# their thermal states, and use ACE to construct the process seen by the spin.
# The observable is the excited-state population $P_e(t)$, compared with the
# isolated result $\sin^2(\Omega t/2)$.
#
# !!! related "Related material"
#     - Tutorial: [Construct a process tensor](@ref)
#     - Theory: [Process Tensors](../theory/process_tensors.md)
#
# !!! script "Companion script"
#     [`scripts/thermal_spinboson_ace.jl`](https://github.com/Gauthameshwar/ProcessTensors.jl/blob/better-docs/scripts/thermal_spinboson_ace.jl)
#     generates the 60-mode, two-temperature comparison below. The model follows
#     Cygorek and Gauger, J. Chem. Phys. **161**, 074111 (2024); the hotter bath
#     is an additional comparison on the same oscillator grid.

# ## Model and spectral density
#
# We set $\hbar=1$, express frequencies in $\mathrm{ps}^{-1}$, and work in the
# rotating frame of a resonant drive. With $A=|e\rangle\langle e|$, the model is
#
# ```math
# H=\Omega S_x+\sum_k\left[
# \omega_k b_k^\dagger b_k+g_k(b_k+b_k^\dagger)A
# +\frac{g_k^2}{\omega_k}A\right].
# ```
#
# The spin starts in $|g\rangle$ (`Dn`); `Up` denotes $|e\rangle$. The bath
# starts uncorrelated with it, with each oscillator thermal under its free
# Hamiltonian. Although the coupling is diagonal in the $g/e$ basis, it does
# not commute with the transverse drive and changes the population dynamics.
#
# !!! note "Why include the counterterm?"
#     Completing the square gives
#     $\omega_k(b_k^\dagger+g_kA/\omega_k)(b_k+g_kA/\omega_k)$, since $A^2=A$.
#     The positive counterterm cancels the static energy lowering
#     $-g_k^2 A/\omega_k$ of a displaced oscillator. It does not remove the
#     bath fluctuations or their dynamical back-action. With a finite Fock
#     cutoff, the displaced oscillator itself is also only approximated.
#
# We use $J(\omega)=0.2\,\omega\exp[-\omega/(3\,\mathrm{ps}^{-1})]$ in the
# convention $J(\omega)=\sum_k g_k^2\delta(\omega-\omega_k)$. Uniform midpoint
# bins give $\omega_k=\omega_{\min}+(k-1/2)\Delta\omega$ and
# $g_k=\sqrt{J(\omega_k)\Delta\omega}$. The bin width belongs in the coupling;
# refining the grid should approximate the same spectral density.

using Logging
using LinearAlgebra
using ITensors
using ITensors.Ops: Trotter
using ProcessTensors

N_bath = 4
local_dim = 3
Ω = 3.0
ω_min, ω_max = 0.0, 30.0
thermal_frequency = 1.0 # θ = k_B T / ħ, in ps^-1, not kelvin
dt, nsteps = 0.20, 12
ace_cutoff, ace_maxdim = 1e-5, 64

spectral_density(ω) = 0.2 * ω * exp(-ω / 3)
Δω = (ω_max - ω_min) / N_bath
frequencies = [ω_min + (k - 0.5) * Δω for k in 1:N_bath]
couplings = sqrt.(spectral_density.(frequencies) .* Δω)
@assert all(>(0), frequencies)

# !!! note "Executable example versus companion figure"
#     These cells use four modes, three levels per mode, and end at $t=2.2$ ps.
#     Their coarse grid starts at $3.75\,\mathrm{ps}^{-1}$ and misses the
#     low-frequency part of the bath. They demonstrate the workflow, not a
#     converged continuum result. The figure uses 60 modes, five levels,
#     $dt=0.05$ ps, $t=5$ ps, and $\theta=1.5,6\,\mathrm{ps}^{-1}$.

# ## Prepare the spin and thermal modes
#
# `thermal_mode` constructs the Gibbs state in the retained oscillator space:
#
# ```math
# \rho_k^{\mathrm{th}}=\frac{1}{Z_{k,M}}
# \sum_{n=0}^{M-1}e^{-n\omega_k/\theta}|n\rangle\langle n|,
# \qquad \theta=k_BT/\hbar.
# ```
#
# The coupling `OpSum` uses site 1 for the oscillator and site 2 for the spin.
# Each mode carries its own counterterm, so it is included exactly once.

system_sites = siteinds("S=1/2", 1)
H_system = OpSum()
H_system += Ω, "Sx", 1
system = spin_system(system_sites, H_system)
initial_density = to_dm(MPS(system_sites, ["Dn"]))

bath_sites = siteinds("Boson", N_bath; dim=local_dim)
bath_liouville_sites = liouv_sites(bath_sites)
modes = BosonicMode[]
for k in 1:N_bath
    ωk, gk = frequencies[k], couplings[k]
    H_mode = OpSum()
    H_mode += ωk, "N", 1
    coupling = OpSum()
    coupling += gk, "A", 1, "ProjUp", 2
    coupling += gk, "Adag", 1, "ProjUp", 2
    coupling += gk^2 / ωk, "ProjUp", 2
    push!(modes, thermal_mode([bath_liouville_sites[k]], H_mode,
                             thermal_frequency; coupling=coupling))
end
# Silence the bath-size warning intended for a joint Dense construction.
bath = with_logger(() -> bosonic_bath(modes), NullLogger())

# ## Construct and evaluate the process
#
# ACE combines the independent mode influences and compresses the temporal
# bonds. Its relative threshold retains singular values $\sigma_i>\epsilon
# \sigma_1$, subject to `maxdim`. The full bath density matrix, with formal
# dimension $M^{2N}$ in Liouville space, is never assembled.

process_tensor = build_process_tensor(
    system; environment=bath, dt=dt, nsteps=nsteps,
    method=ACE(cutoff=ace_cutoff, maxdim=ace_maxdim),
    sys_alg=Trotter{2}(), combine_alg=Trotter{2}(), progress=false,
)
trajectory = evolve(process_tensor, initial_density);
println((element_type=eltype(trajectory.states_liouville),))
println(sprint(show, foldl(*, last(trajectory.states_liouville))))

# For this one-spin output, the excited population is the first diagonal entry
# divided by the trace. The helper is reused at every time; raw trace drift is
# reported separately. No density-matrix eigenvalues are clipped.

function one_spin_matrix(ρ)
    tensor = foldl(*, ρ)
    site = only(filter(i -> plev(i) == 0 && hastags(i, "Site"), inds(tensor)))
    return ComplexF64.(Array(tensor, prime(site), site))
end

states = one_spin_matrix.(trajectory.states_hilbert)
traces = tr.(states)
@assert all(z -> isfinite(z) && abs(z) > 1e-12, traces)
population = [real(ρ[1, 1] / z) for (ρ, z) in zip(states, traces)]
isolated_population = sin.(Ω .* trajectory.times ./ 2) .^ 2
max_trace_error = maximum(abs.(traces .- 1))
println((final_time=last(trajectory.times), final_population=last(population),
         isolated_final_population=last(isolated_population),
         max_trace_error=max_trace_error, max_pt_bond=maxlinkdim(process_tensor)))
@assert all(p -> isfinite(p) && -2e-3 <= p <= 1 + 2e-3, population)
@assert max_trace_error < 2e-3

# ## Interpret the loss of Rabi contrast
#
# ![Ohmic spectral density and thermal damping of driven population oscillations](../assets/examples/thermal_spinboson_ace.png)
#
# The upper panel shows where the continuum is sampled. Most spectral weight
# lies near the exponential cutoff scale; the midpoint grid also resolves the
# low-frequency modes that the four-mode example misses.
#
# In the lower panel, the isolated spin repeatedly reaches populations zero
# and one, with period $2\pi/\Omega\simeq2.09$ ps. The bath-coupled curves
# develop lower peaks and higher troughs: population transfer becomes less
# complete on successive cycles. The hotter curve has a smaller first peak
# and is closer to $1/2$ by the end of the plotted interval.
#
# The $g/e$ components displace the oscillators differently, allowing the bath
# to retain information about the system's history. This affects the coherence
# sustaining the driven rotations. Temperature changes the initial oscillator
# fluctuations; here it changes the damping even though $J(\omega)$ and the
# couplings are held fixed. This is a comparison over the displayed time window,
# not a general assertion that every hotter bath damps every system faster.
#
# !!! note "Equal populations do not establish thermalisation"
#     $P_e\simeq1/2$ specifies one observable. It neither shows that the spin is
#     maximally mixed nor establishes a Gibbs state: off-diagonal coherence may
#     remain. A finite, discretised bath can also exhibit later recurrences.
#
# ### Resolve the thermal oscillator space
#
# The local cutoff is especially important when $\theta/\omega_k$ is large.
# For an untruncated free oscillator, the total initial Gibbs probability above
# the retained levels is $\Pr(n\ge M)=e^{-M\omega_k/\theta}$. It is a useful
# preparation diagnostic, not a bound on the eventual population error.

thermal_tail = thermal_frequency == 0 ? zeros(N_bath) :
               exp.(-local_dim .* frequencies ./ thermal_frequency)
println((largest_omitted_Gibbs_weight=maximum(thermal_tail),))

# At the companion grid's lowest frequency $\omega_1=0.25\,\mathrm{ps}^{-1}$,
# five levels omit about 0.43 of the infinite-oscillator Gibbs weight at
# $\theta=1.5$, and 0.81 at $\theta=6$. The simulated truncated Gibbs states
# are normalised, but those numbers make local-dimension convergence essential
# before interpreting the curves quantitatively as a continuum thermal bath.
# Bath-induced displacement can require additional levels even at zero temperature.
# Timestep, frequency window/grid, and ACE truncation must also be converged;
# a small trace error or an unreached bond cap does not settle these questions.
#
# !!! tip "Try changing"
#     These edits test which changes in Rabi contrast are physical and which
#     come from discretisation or compression.
#
#     - Increase `local_dim` at fixed mode grid and temperature, particularly
#       for the hotter bath. Do the peak heights and troughs stabilise?
#     - Try allocating more ACE modes where $|J(\omega)|$ is largest, and using 
#       fewer modes where $|J(\omega)|$ is small. Does concentrating frequency 
#       samples this way affect the Rabi oscillations?
