# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors #src
# SPDX-License-Identifier: MIT #src
# #src
# File: docs/literate/examples/laser_driven_tdvp.jl #src
# Contributor: Gauthameshwar S. #src
# #src
# Demonstrates midpoint two-site TDVP for an interacting spin chain driven by #src
# a time-dependent Gaussian laser pulse. #src

# # Laser-driven TDVP dynamics
#
# This example shows how an ordinary Julia function can represent a
# time-dependent Hamiltonian. At each timestep, we evaluate that function at
# the interval midpoint, construct an MPO, and pass it to the usual TDVP
# routine. 
#
# The complete plotting workflow is available in
# `scripts/laser_driven_tdvp.jl`.

# ## Laser-driven spin chain
#
# We consider a closed interacting spin chain driven by a Gaussian pulse,
#
# ```math
# H(t)
# =
# -J\sum_{j=1}^{N-1} Z_j Z_{j+1}
# -\frac{\Delta}{2}\sum_{j=1}^{N} Z_j
# +\frac{\Omega(t)}{2}\sum_{j=1}^{N} X_j ,
# ```
#
# where
#
# ```math
# \Omega(t)
# =
# \Omega_0
# \exp\left[-\frac{(t-t_c)^2}{2\sigma^2}\right].
# ```
#
# The longitudinal field sets the detuning, the Gaussian transverse field
# drives coherent spin flips, and the Ising interaction makes the response
# genuinely many-body. Starting from
# ``|\psi(0)\rangle=|\downarrow\cdots\downarrow\rangle``, we monitor the
# instantaneous energy density and the mean excitation density
#
# ```math
# \bar n(t)=\frac{1}{N}\sum_j
# \left\langle\frac{I+Z_j}{2}\right\rangle .
# ```

# ### Model and parameters

using ITensors
using ProcessTensors
using Statistics: mean

const N = 6
const J = 0.4
const detuning = 1.0
const Ω0 = 2.5
const pulse_center = 1.2
const pulse_width = 0.35
const dt = 0.1
const final_time = 2.4
const maxdim = 60
const cutoff = 1e-10

sites = siteinds("S=1/2", N)
initial_state = MPS(sites, fill("Dn", N))

# The pulse envelope and Hamiltonian are ordinary Julia functions. Calling
# `laser_driven_hamiltonian(t)` produces the `OpSum` at time `t`.

gaussian_drive(t::Real) =
    Ω0 * exp(-((t - pulse_center)^2) / (2pulse_width^2))

function laser_driven_hamiltonian(t::Real)
    H = OpSum()
    for j in 1:(N - 1)
        H += -J, "Z", j, "Z", j + 1
    end
    for j in 1:N
        H += -detuning / 2, "Z", j
        H += gaussian_drive(t) / 2, "X", j
    end
    return H
end

H_at_pulse_center = MPO(laser_driven_hamiltonian(pulse_center), sites)
@assert length(H_at_pulse_center) == N

# ## Midpoint TDVP evolution
#
# On an interval ``[t_n,t_n+\Delta t]``, the midpoint approximation uses
#
# ```math
# H_{n+\frac12}=H\left(t_n+\frac{\Delta t}{2}\right)
# ```
#
# for one ordinary TDVP step,
#
# ```math
# |\psi(t_{n+1})\rangle
# \approx
# \exp[-i\Delta t\,H_{n+\frac12}]|\psi(t_n)\rangle .
# ```
#
# We use two-site TDVP (`nsite=2`) so that the MPS bond dimensions can grow as
# the drive and interactions generate entanglement. The core workflow is the
# midpoint `OpSum` → `MPO` → `tdvp` sequence visible inside this loop.
# 
# !!! note "Time-dependent Hamiltonians in ITensorMPS TDVP"
#     In earlier versions of ITensorMPS, time-dependent Hamiltonians were defined
#     using the `TimeDependentHamiltonian` type. This method is now deprecated.
#     The current recommended approach is to define an ordinary Julia function that
#     returns an `OpSum` (or `MPO`) for a given time. This function is then called
#     at each time step to construct the appropriate Hamiltonian for the TDVP evolution.

nsteps = round(Int, final_time / dt)
times = collect(range(0.0; step=dt, length=nsteps + 1))

trajectory = let
    ψ = copy(initial_state)
    energies = Float64[]
    excitations = Float64[]
    norm_errors = Float64[]

    for step in eachindex(times)
        H_now = MPO(laser_driven_hamiltonian(times[step]), sites)
        z_values = real.(expect(ψ, "Z"))

        push!(energies, real(inner(ψ', H_now, ψ)) / N)
        push!(excitations, mean((1 .+ z_values) ./ 2))
        push!(norm_errors, abs(real(inner(ψ, ψ)) - 1))

        step == length(times) && continue

        t_mid = times[step] + dt / 2
        H_mid = MPO(laser_driven_hamiltonian(t_mid), sites)
        ψ = tdvp(
            H_mid,
            -1im * dt,
            ψ;
            time_step=-1im * dt,
            nsite=2,
            maxdim=maxdim,
            cutoff=cutoff,
            outputlevel=0,
        )
    end

    (
        final_state=ψ,
        energy_density=energies,
        excitation_density=excitations,
        norm_error=norm_errors,
    )
end

@assert maximum(trajectory.norm_error) < 1e-8
@assert all(x -> -1e-10 ≤ x ≤ 1 + 1e-10, trajectory.excitation_density)
@assert all(isfinite, trajectory.energy_density)

# ## Physical response
#
# The energy is not conserved because the external pulse performs work on the
# chain. The excitation density measures the coherent population transferred
# away from the initial unexcited product state. The shaded interval in the
# script-generated figure marks one pulse width on either side of the pulse
# center. The plotting script uses a longer ``N=12`` run than the compact
# executable example above, but follows exactly the same midpoint workflow.
#
# ![Instantaneous energy and excitation density for the laser-driven chain](../assets/examples/laser_driven_tdvp.png)
#
# !!! note "Numerical accuracy"
#     The midpoint approximation must resolve the pulse, so `dt` should be much
#     smaller than `pulse_width`. Two-site TDVP also introduces MPS truncation
#     error controlled by `cutoff` and `maxdim`; the norm check above provides a
#     compact diagnostic for this unitary evolution.
#
# !!! summary "Example takeaways"
#     - A Julia function returning an `OpSum` is enough to define ``H(t)``.
#     - Construct the midpoint MPO before each ordinary two-site TDVP step.
#     - For a driven system, instantaneous energy and excitation density expose
#       complementary effects of the applied pulse.
#     - Run `scripts/laser_driven_tdvp.jl` to regenerate the figure.
