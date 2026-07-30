# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: scripts/laser_driven_tdvp.jl
# Contributor: Gauthameshwar S.
#
# Evolves an interacting spin chain under a Gaussian laser pulse with midpoint
# two-site TDVP and plots its energy and excitation response.
#
# Run with:
#   julia --project=. scripts/laser_driven_tdvp.jl

import Pkg

const REPO_ROOT = dirname(@__DIR__)
const _PLOT_ENV = joinpath(@__DIR__, ".plot_examples_env")

function activate_plot_examples_env!()
    mkpath(_PLOT_ENV)
    Pkg.activate(_PLOT_ENV)
    manifest = joinpath(_PLOT_ENV, "Manifest.toml")
    if !isfile(manifest)
        Pkg.develop(Pkg.PackageSpec(path=REPO_ROOT))
        Pkg.add(Pkg.PackageSpec(name="CairoMakie"))
    else
        Pkg.resolve()
        Pkg.instantiate()
    end
    return nothing
end

activate_plot_examples_env!()

using CairoMakie
using ITensors
using ITensorMPS: expect, maxlinkdim, tdvp
using ProcessTensors
using Statistics: mean

CairoMakie.activate!()

# User-adjustable parameters
const N = 12
const J = 0.4
const detuning = 1.0
const Ω0 = 2.5
const pulse_center = 3.0
const pulse_width = 0.7
const dt = 0.1
const final_time = 6.0
const maxdim = 100
const cutoff = 1e-10

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

nsteps = round(Int, final_time / dt)
isapprox(nsteps * dt, final_time; atol=100eps(Float64)) ||
    throw(ArgumentError("final_time must be an integer multiple of dt."))

times = collect(range(0.0; step=dt, length=nsteps + 1))
sites = siteinds("S=1/2", N)
initial_state = MPS(sites, fill("Dn", N))

energy_density = Vector{Float64}(undef, length(times))
excitation_density = Vector{Float64}(undef, length(times))
norm_error = Vector{Float64}(undef, length(times))
bond_dimensions = Vector{Int}(undef, length(times))

function record_observables!(step::Int, ψ)
    H_now = MPO(laser_driven_hamiltonian(times[step]), sites)
    z_values = real.(expect(ψ, "Z"))
    energy_density[step] = real(inner(ψ', H_now, ψ)) / N
    excitation_density[step] = mean((1 .+ z_values) ./ 2)
    norm_error[step] = abs(real(inner(ψ, ψ)) - 1)
    bond_dimensions[step] = maxlinkdim(ψ)
    return nothing
end

println("Laser-driven TDVP: N=$N, dt=$dt, final_time=$final_time")
final_state = let ψ = copy(initial_state)
    record_observables!(1, ψ)
    status_stride = max(1, nsteps ÷ 10)

    for step in 1:nsteps
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

        record_observables!(step + 1, ψ)
        if step == 1 || step == nsteps || step % status_stride == 0
            print(
                "\rTDVP step $(lpad(step, 3))/$nsteps" *
                "  t=$(round(times[step + 1]; digits=2))" *
                "  max bond=$(bond_dimensions[step + 1])",
            )
            flush(stdout)
        end
    end
    ψ
end
println()

maximum(norm_error) < 1e-8 ||
    @warn "TDVP norm drift exceeded tolerance." maximum_norm_error=maximum(norm_error)
all(x -> -1e-10 ≤ x ≤ 1 + 1e-10, excitation_density) ||
    @warn "Excitation density left its physical interval."

figure = Figure(size=(900, 680))

energy_axis = Axis(
    figure[1, 1];
    xlabel="time",
    ylabel="⟨H(t)⟩ / N",
    title="Instantaneous energy density",
)
lines!(energy_axis, times, energy_density; linewidth=2.5, color=:royalblue)
vspan!(
    energy_axis,
    pulse_center - pulse_width,
    pulse_center + pulse_width;
    color=(:orange, 0.12),
)

excitation_axis = Axis(
    figure[2, 1];
    xlabel="time",
    ylabel="mean excitation density",
    title="Coherent excitation response",
)
lines!(
    excitation_axis,
    times,
    excitation_density;
    linewidth=2.5,
    color=:darkred,
)
vspan!(
    excitation_axis,
    pulse_center - pulse_width,
    pulse_center + pulse_width;
    color=(:orange, 0.12),
)
ylims!(excitation_axis, -0.02, 1.02)

output_dir = joinpath(@__DIR__, "figures")
mkpath(output_dir)
figure_path = joinpath(output_dir, "laser_driven_tdvp.png")
save(figure_path, figure; px_per_unit=2)

println("Completed laser-driven TDVP evolution.")
println("  maximum norm error: $(maximum(norm_error))")
println("  maximum bond dimension: $(maximum(bond_dimensions))")
println("  saved figure: $figure_path")
