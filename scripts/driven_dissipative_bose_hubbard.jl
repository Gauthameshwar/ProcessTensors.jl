# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: scripts/driven_dissipative_bose_hubbard.jl
# Contributor: Gauthameshwar S.
#
# Evolves a coherently pumped Bose–Hubbard chain with local loss for several
# onsite interactions and plots the common pump with the mean-occupation response.
#
# Run with:
#   julia --project=. scripts/driven_dissipative_bose_hubbard.jl

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
using ProcessTensors

CairoMakie.activate!()

# User-adjustable parameters
const N = 5
const local_dim = 5
const hopping = 0.2
const detuning = 0.0
const pump_strength = 0.30
const ramp_time = 0.6
const loss_rate = 0.8
const interaction_strengths = [0.0, 0.75, 2.5]
const dt = 0.1
const final_time = 7.0
const maxdim = 50
const cutoff = 1e-10
const trace_warning_tolerance = 1e-3

pump_amplitude(t::Real) =
    pump_strength * (1 - exp(-(t^2) / (2 * ramp_time^2)))

function driven_bose_hubbard_hamiltonian(t::Real, interaction::Real)
    pump = pump_amplitude(t)
    H = OpSum()
    for j in 1:(N - 1)
        H += -hopping, "Adag", j, "A", j + 1
        H += -hopping, "A", j, "Adag", j + 1
    end
    for j in 1:N
        H += -detuning, "N", j
        H += interaction / 2, "N", j, "N", j
        H += -interaction / 2, "N", j
        H += pump, "A", j
        H += pump, "Adag", j
    end
    return H
end

nsteps = round(Int, final_time / dt)
isapprox(nsteps * dt, final_time; atol=100eps(Float64)) ||
    throw(ArgumentError("final_time must be an integer multiple of dt."))

times = collect(range(0.0; step=dt, length=nsteps + 1))
physical_sites = siteinds("Boson", N; dim=local_dim, conserve_qns=false)
liouville_sites = liouv_sites(physical_sites)
loss_jump_operators = [(loss_rate, "A", j) for j in 1:N]

mean_occupation_operator = let observable = OpSum()
    for j in 1:N
        observable += 1 / N, "N", j
    end
    observable
end
mean_occupation_liouville = to_liouville(
    MPO(mean_occupation_operator, physical_sites);
    sites=liouville_sites,
)

initial_density_liouville = to_liouville(
    to_dm(MPS(physical_sites, fill("0", N)));
    sites=liouville_sites,
)

println("Driven-dissipative Bose–Hubbard")
println("  N=$N, local_dim=$local_dim, dt=$dt, final_time=$final_time")
println("  U ∈ $(interaction_strengths)")

occupation_curves = Dict{Float64,Vector{Float64}}()
status_stride = max(1, nsteps ÷ 10)

for interaction in interaction_strengths
    println()
    println("Running U = $interaction ...")

    occupations = Vector{Float64}(undef, length(times))
    trace_errors = Vector{Float64}(undef, length(times))
    bond_dimensions = Vector{Int}(undef, length(times))

    density = copy(initial_density_liouville)
    for step in eachindex(times)
        density_trace = tr(to_hilbert(density))
        occupations[step] =
            real(inner(mean_occupation_liouville, density) / density_trace)
        trace_errors[step] = abs(density_trace - 1)
        bond_dimensions[step] = maxlinkdim(density)

        step == length(times) && continue

        midpoint = times[step] + dt / 2
        L_mid = liouvillian_mpo(
            driven_bose_hubbard_hamiltonian(midpoint, interaction),
            liouville_sites;
            jump_ops=loss_jump_operators,
        )
        density = tdvp(
            L_mid,
            dt,
            density;
            time_step=dt,
            nsite=2,
            maxdim=maxdim,
            cutoff=cutoff,
            outputlevel=0,
        )

        if step == 1 || step == nsteps || step % status_stride == 0
            print(
                "\r  TDVP step $(lpad(step, 3))/$nsteps" *
                "  t=$(round(times[step + 1]; digits=2))" *
                "  n̄=$(round(occupations[step]; digits=4))" *
                "  bond=$(bond_dimensions[step])",
            )
            flush(stdout)
        end
    end
    println()

    max_trace_error = maximum(trace_errors)
    max_trace_error ≤ trace_warning_tolerance ||
        @warn "Trace drift exceeded tolerance." interaction=interaction max_trace_error=max_trace_error
    all(n -> -1e-8 ≤ n ≤ local_dim - 1 + 1e-8, occupations) ||
        @warn "Mean occupation left its physical interval." interaction=interaction

    occupation_curves[interaction] = occupations
    println("  final n̄ = $(occupations[end])")
    println("  max trace error = $max_trace_error")
    println("  max bond dim = $(maximum(bond_dimensions))")
end

pump_values = pump_amplitude.(times)

figure = Figure(size=(900, 680))

pump_axis = Axis(
    figure[1, 1];
    xlabel="time",
    ylabel="F(t)",
    title="Pump amplitude",
)
lines!(pump_axis, times, pump_values; linewidth=2.5, color=:royalblue)
ylims!(pump_axis, -0.02, pump_strength * 1.15)

occupation_axis = Axis(
    figure[2, 1];
    xlabel="time",
    ylabel="mean occupation n̄(t)",
    title="Interaction dependence under the same pump and loss",
)
colors = [:black, :darkorange, :firebrick]
for (interaction, color) in zip(interaction_strengths, colors)
    lines!(
        occupation_axis,
        times,
        occupation_curves[interaction];
        linewidth=2.5,
        color=color,
        label="U = $interaction",
    )
end
axislegend(occupation_axis; position=:rb)

output_dir = joinpath(@__DIR__, "figures")
mkpath(output_dir)
figure_path = joinpath(output_dir, "driven_dissipative_bose_hubbard.png")
save(figure_path, figure; px_per_unit=2)

println()
println("Completed driven-dissipative Bose–Hubbard comparisons.")
println("  saved figure: $figure_path")
