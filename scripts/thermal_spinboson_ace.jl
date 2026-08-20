# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: scripts/thermal_spinboson_ace.jl
# Contributor: Gauthameshwar S.
#
# Reproduces the thermal spin-boson example of:
# M. Cygorek and E. M. Gauger, J. Chem. Phys. 161, 074111 (2024), Fig. 3(c,f),
# and compares the published bath temperature with a hotter Gibbs state on the
# same oscillator grid.
#
# Published parameters reproduced here:
#   dt = 0.05 ps
#   t_final = 5 ps
#   ACE threshold epsilon = 1e-5
#   Omega = 3 ps^-1
#   N_bath = 60
#   local boson Hilbert dimension M = 5
#   omega in [0, 30] ps^-1
#   k_B T / hbar: thermal_frequency and hot_thermal_frequency (ps^-1)
#   J(omega) = 0.2 omega exp[-omega/(3 ps^-1)]
#
# Run with:
#   julia --project=. scripts/thermal_spinboson_ace.jl

import Pkg

const REPO_ROOT = dirname(@__DIR__)
const _PLOT_ENV = joinpath(@__DIR__, ".plot_examples_env")

function activate_plot_examples_env!()
    mkpath(_PLOT_ENV)
    Pkg.activate(_PLOT_ENV)
    manifest = joinpath(_PLOT_ENV, "Manifest.toml")

    if !isfile(manifest)
        Pkg.develop(Pkg.PackageSpec(path=REPO_ROOT))
        Pkg.add([
            Pkg.PackageSpec(name="CairoMakie"),
            Pkg.PackageSpec(name="LaTeXStrings"),
        ])
    else
        Pkg.resolve()
        Pkg.instantiate()
    end
    return nothing
end

activate_plot_examples_env!()

using Logging
using Printf
using CairoMakie
using ITensors
using ITensors.Ops: Trotter
using LaTeXStrings
using ProcessTensors

CairoMakie.activate!()

# ------------------------------------------------------------------------------
# 1. Small utilities
# ------------------------------------------------------------------------------

const STATUS_WIDTH = 100

function print_section(title::AbstractString)
    println()
    println(title)
    println("-"^length(title))
end

function update_status(message::AbstractString)
    print("\r", rpad(message, STATUS_WIDTH))
    flush(stdout)
end

function finish_status(message::AbstractString="")
    print("\r", " "^STATUS_WIDTH, "\r")
    isempty(message) || println(message)
    flush(stdout)
end

ohmic_spectral_density(ω) = 0.2 * ω * exp(-ω / 3)

ace_T_label(T) = @sprintf("T=%g (ACE)", T)

function uniform_mode_grid(N_bath, ω_min, ω_max)
    Δω = (ω_max - ω_min) / N_bath
    frequencies = [ω_min + (k - 0.5) * Δω for k in 1:N_bath]
    spacings = fill(Δω, N_bath)
    couplings = sqrt.(ohmic_spectral_density.(frequencies) .* spacings)
    return frequencies, spacings, couplings
end

function thermal_spinboson_bath(frequencies, couplings, thermal_frequency)
    N_modes = length(frequencies)
    bath_sites = siteinds("Boson", N_modes; dim=local_dim)
    bath_liouville_sites = liouv_sites(bath_sites)
    modes = BosonicMode[]

    for k in 1:N_modes
        update_status(
            @sprintf(
                "  mode %2d / %d   omega=%6.3f   g=%8.5f",
                k,
                N_modes,
                frequencies[k],
                couplings[k],
            ),
        )

        ωk = frequencies[k]
        gk = couplings[k]

        mode_hamiltonian = OpSum()
        mode_hamiltonian += ωk, "N", 1

        mode_coupling = OpSum()
        mode_coupling += gk, "A", 1, "ProjUp", 2
        mode_coupling += gk, "Adag", 1, "ProjUp", 2
        mode_coupling += gk^2 / ωk, "ProjUp", 2

        initial_mode_density = thermal_boson_density(
            bath_sites[k],
            bath_liouville_sites[k],
            ωk,
            thermal_frequency,
            local_dim,
        )

        push!(
            modes,
            bosonic_mode(
                [bath_liouville_sites[k]],
                mode_hamiltonian,
                initial_mode_density;
                coupling=mode_coupling,
            ),
        )
    end

    finish_status("  prepared $N_modes thermal oscillator modes")
    return with_logger(NullLogger()) do
        bosonic_bath(modes)
    end
end

function thermal_boson_density(
    physical_site,
    liouville_site,
    ω,
    thermal_frequency,
    local_dim,
)
    occupations = 0:(local_dim - 1)
    weights = exp.(-ω .* occupations ./ thermal_frequency)
    weights ./= sum(weights)

    number_states = [
        MPS([physical_site], [string(n)])
        for n in occupations
    ]
    density = to_dm(number_states; coeffs=weights)

    return to_liouville(
        density;
        sites=[liouville_site],
    )
end

function one_site_density_matrix(ρ)
    tensor = foldl(*, ρ)
    site = only(
        filter(
            index -> plev(index) == 0 && hastags(index, "Site"),
            inds(tensor),
        ),
    )
    return ComplexF64.(Array(tensor, prime(site), site))
end

function excited_population(trajectory, projector)
    values = Float64[]
    trace_errors = Float64[]

    for ρ in trajectory.states_hilbert
        ρ_matrix = one_site_density_matrix(ρ)
        trace_value = tr(ρ_matrix)

        push!(
            values,
            real(tr(projector * ρ_matrix) / trace_value),
        )
        push!(trace_errors, abs(trace_value - 1))
    end

    return values, trace_errors
end

# ------------------------------------------------------------------------------
# 2. Published parameters
# ------------------------------------------------------------------------------

const N_bath = 60
const local_dim = 5

const Ω = 3.0
const ω_min = 0.0
const ω_max = 30.0
const thermal_frequency = 1.5 # k_B T / hbar, ps^-1 (published)
const hot_thermal_frequency = 6.0 # k_B T / hbar, ps^-1

const dt = 0.05
const final_time = 5.0
const nsteps = round(Int, final_time / dt) + 1

const ace_cutoff = 1e-5
const ace_maxdim = 512

const trace_warning_tolerance = 1e-5
const population_tolerance = 2e-3

output_dir = joinpath(@__DIR__, "figures")
mkpath(output_dir)
figure_path = joinpath(output_dir, "thermal_spinboson_ace.png")

frequencies, spacings, couplings =
    uniform_mode_grid(N_bath, ω_min, ω_max)

@assert all(>(0), frequencies)
@assert all(isfinite, couplings)
@assert isapprox((nsteps - 1) * dt, final_time; atol=100eps(Float64))

# ------------------------------------------------------------------------------
# 3. Physical setup
# ------------------------------------------------------------------------------

print_section("Thermal spin-boson ACE benchmark")

println("Reproducing the driven Ohmic-bath example of Cygorek & Gauger.")
@printf("  drive Omega:                    %.3f ps^-1\n", Ω)
@printf("  bath modes:                     %d\n", N_bath)
@printf("  local boson dimension M:        %d\n", local_dim)
@printf("  omega window:                   [%.1f, %.1f] ps^-1\n", ω_min, ω_max)
@printf("  uniform spacing:                %.3f ps^-1\n", first(spacings))
@printf("  k_B T / hbar:                   %.3f and %.3f ps^-1\n", thermal_frequency, hot_thermal_frequency)
@printf("  dt:                             %.3f ps\n", dt)
@printf("  final time:                     %.1f ps\n", final_time)
@printf("  ACE threshold epsilon:          %.1e\n", ace_cutoff)
@printf("  ACE maxdim safety cap:          %d\n", ace_maxdim)
println("  spectral density:               J(w) = 0.2 w exp(-w/3)")
println("  system-bath operator:           |e><e| (b + b†)")
println("  polaron-shift subtraction:      enabled")

system_sites = siteinds("S=1/2", 1)

system_hamiltonian = OpSum()
system_hamiltonian += Ω, "Sx", 1

system = spin_system(system_sites, system_hamiltonian)
initial_density = to_dm(MPS(system_sites, ["Dn"]))

excited_projector = ComplexF64.(
    Array(
        op("ProjUp", system_sites[1]),
        prime(system_sites[1]),
        system_sites[1],
    ),
)

# ------------------------------------------------------------------------------
# 4. Closed-system reference
# ------------------------------------------------------------------------------

print_section("Closed Rabi reference")

update_status("  building trivial process tensor")
closed_build_time = @elapsed begin
    closed_process = build_process_tensor(
        system;
        dt=dt,
        nsteps=nsteps,
        sys_alg=Trotter{2}(),
        progress=false,
    )
end
finish_status(@sprintf("  trivial PT built in %.3f s", closed_build_time))

update_status("  evolving isolated two-level system")
closed_evolution_time = @elapsed begin
    closed_trajectory = evolve(closed_process, initial_density)
end
finish_status(
    @sprintf(
        "  isolated trajectory evolved in %.3f s",
        closed_evolution_time,
    ),
)

closed_population, closed_trace_errors =
    excited_population(closed_trajectory, excited_projector)

analytical_population =
    sin.(Ω .* closed_trajectory.times ./ 2) .^ 2
closed_analytic_error =
    maximum(abs.(closed_population .- analytical_population))

@printf(
    "  max |P_e - sin^2(Omega t/2)|:  %.3e\n",
    closed_analytic_error,
)

# ------------------------------------------------------------------------------
# 5. Thermal baths on the same oscillator grid
# ------------------------------------------------------------------------------

print_section("Preparing 60-mode thermal baths")

bath_T1 = thermal_spinboson_bath(frequencies, couplings, thermal_frequency)
bath_T3 = thermal_spinboson_bath(frequencies, couplings, hot_thermal_frequency)

formal_hilbert_dimension = BigInt(local_dim)^N_bath
formal_liouville_dimension = BigInt(local_dim)^(2N_bath)

println("Formal environment sizes after local truncation:")
println("  bath Hilbert dimension:        $formal_hilbert_dimension")
println("  bath Liouville dimension:      $formal_liouville_dimension")

# ------------------------------------------------------------------------------
# 6. ACE process tensors
# ------------------------------------------------------------------------------

print_section(@sprintf("Building ACE process tensor at T=%g", thermal_frequency))

println("Live mode-join progress is reported by ProcessTensors.jl.")
ace_build_time_T1 = @elapsed begin
    process_tensor_T1 = build_process_tensor(
        system;
        method=ACE(
            cutoff=ace_cutoff,
            maxdim=ace_maxdim,
        ),
        environment=bath_T1,
        dt=dt,
        nsteps=nsteps,
        sys_alg=Trotter{2}(),
        combine_alg=Trotter{2}(),
        progress=true,
        verbose=false,
    )
end

max_pt_bond_T1 = maxlinkdim(process_tensor_T1)

println()
println(@sprintf("ACE build complete at T=%g.", thermal_frequency))
@printf("  build time:                     %.3f s\n", ace_build_time_T1)
@printf("  maximum retained PT bond:       %d\n", max_pt_bond_T1)

print_section(@sprintf("Building ACE process tensor at T=%g", hot_thermal_frequency))

println("Live mode-join progress is reported by ProcessTensors.jl.")
ace_build_time_T3 = @elapsed begin
    process_tensor_T3 = build_process_tensor(
        system;
        method=ACE(
            cutoff=ace_cutoff,
            maxdim=ace_maxdim,
        ),
        environment=bath_T3,
        dt=dt,
        nsteps=nsteps,
        sys_alg=Trotter{2}(),
        combine_alg=Trotter{2}(),
        progress=true,
        verbose=false,
    )
end

max_pt_bond_T3 = maxlinkdim(process_tensor_T3)

println()
println(@sprintf("ACE build complete at T=%g.", hot_thermal_frequency))
@printf("  build time:                     %.3f s\n", ace_build_time_T3)
@printf("  maximum retained PT bond:       %d\n", max_pt_bond_T3)

if max(max_pt_bond_T1, max_pt_bond_T3) >= ace_maxdim
    @warn(
        "ACE reached the maxdim safety cap.",
        max_pt_bond_T1=max_pt_bond_T1,
        max_pt_bond_T3=max_pt_bond_T3,
        ace_maxdim=ace_maxdim,
    )
end

# ------------------------------------------------------------------------------
# 7. Reduced-system dynamics
# ------------------------------------------------------------------------------

print_section("Evaluating open-system trajectories")

update_status(@sprintf("  evolving ACE process tensor at T=%g", thermal_frequency))
open_evolution_time_T1 = @elapsed begin
    open_trajectory_T1 = evolve(process_tensor_T1, initial_density)
end
finish_status(
    @sprintf(
        "  T=%g trajectory evolved in %.3f s",
        thermal_frequency,
        open_evolution_time_T1,
    ),
)

update_status(@sprintf("  evolving ACE process tensor at T=%g", hot_thermal_frequency))
open_evolution_time_T3 = @elapsed begin
    open_trajectory_T3 = evolve(process_tensor_T3, initial_density)
end
finish_status(
    @sprintf(
        "  T=%g trajectory evolved in %.3f s",
        hot_thermal_frequency,
        open_evolution_time_T3,
    ),
)

open_population_T1, open_trace_errors_T1 =
    excited_population(open_trajectory_T1, excited_projector)
open_population_T3, open_trace_errors_T3 =
    excited_population(open_trajectory_T3, excited_projector)

max_trace_error = maximum((
    maximum(closed_trace_errors),
    maximum(open_trace_errors_T1),
    maximum(open_trace_errors_T3),
))

@assert all(isfinite, open_population_T1)
@assert all(isfinite, open_population_T3)
@assert all(isfinite, closed_population)
@assert all(
    value -> -population_tolerance <= value <= 1 + population_tolerance,
    open_population_T1,
)
@assert all(
    value -> -population_tolerance <= value <= 1 + population_tolerance,
    open_population_T3,
)

if max_trace_error > trace_warning_tolerance
    @warn(
        "Trace drift exceeds the preferred script tolerance.",
        max_trace_error=max_trace_error,
        trace_warning_tolerance=trace_warning_tolerance,
    )
end

println("Trajectory diagnostics")
@printf("  max trace error:                %.3e\n", max_trace_error)
@printf("  final P_e, isolated:            %.6f\n", closed_population[end])
@printf("  final P_e, %s:            %.6f\n", ace_T_label(thermal_frequency), open_population_T1[end])
@printf("  final P_e, %s:            %.6f\n", ace_T_label(hot_thermal_frequency), open_population_T3[end])
@printf("  closed evolution time:          %.3f s\n", closed_evolution_time)
@printf("  %s evolution time:         %.3f s\n", ace_T_label(thermal_frequency), open_evolution_time_T1)
@printf("  %s evolution time:         %.3f s\n", ace_T_label(hot_thermal_frequency), open_evolution_time_T3)

# ------------------------------------------------------------------------------
# 8. Figure
# ------------------------------------------------------------------------------

print_section("Plotting")

isolated_color = :gray35
T1_color = :steelblue
T3_color = :darkorange
ace_fill = :lightskyblue

figure = Figure(size=(1100, 850))

spectral_axis = Axis(
    figure[1, 1];
    xlabel=L"\omega\;(\mathrm{ps}^{-1})",
    ylabel=L"J(\omega)",
    title="Ohmic environment and its 60-mode discretization",
)

ω_plot = range(ω_min, ω_max; length=800)
lines!(
    spectral_axis,
    ω_plot,
    ohmic_spectral_density.(ω_plot);
    color=T1_color,
    linewidth=2.6,
    label=L"J(\omega)=0.2\,\omega e^{-\omega/3}",
)
scatter!(
    spectral_axis,
    frequencies,
    ohmic_spectral_density.(frequencies);
    marker=:circle,
    markersize=12,
    color=ace_fill,
    strokecolor=:black,
    strokewidth=2.4,
    label="ACE modes",
)
axislegend(spectral_axis; position=:rt)

population_axis = Axis(
    figure[2, 1];
    xlabel=L"t\;(\mathrm{ps})",
    ylabel=L"P_e(t)",
    title=L"Thermal spin-boson damping ($k_{\mathrm{B}}T/\hbar$ in $\mathrm{ps}^{-1}$)",
)

lines!(
    population_axis,
    closed_trajectory.times,
    closed_population;
    color=isolated_color,
    linewidth=2.5,
    linestyle=:dash,
    label="isolated TLS",
)
lines!(
    population_axis,
    open_trajectory_T1.times,
    open_population_T1;
    color=T1_color,
    linewidth=2.6,
    label=ace_T_label(thermal_frequency),
)
lines!(
    population_axis,
    open_trajectory_T3.times,
    open_population_T3;
    color=T3_color,
    linewidth=2.6,
    label=ace_T_label(hot_thermal_frequency),
)

ylims!(population_axis, -0.03, 1.03)
axislegend(population_axis; position=:rt)

rowgap!(figure.layout, 18)
save(figure_path, figure)

println("Saved figure:")
println("  $figure_path")

# ------------------------------------------------------------------------------
# 9. Final summary
# ------------------------------------------------------------------------------

print_section("Summary")

println("Completed thermal spin-boson ACE example.")
println("  figure:                         $figure_path")
@printf("  bath modes:                     %d\n", N_bath)
@printf("  %s χmax:                   %d\n", ace_T_label(thermal_frequency), max_pt_bond_T1)
@printf("  %s χmax:                   %d\n", ace_T_label(hot_thermal_frequency), max_pt_bond_T3)
@printf("  maximum trace error:            %.3e\n", max_trace_error)
@printf("  closed analytic error:          %.3e\n", closed_analytic_error)
@printf("  %s build time:             %.3f s\n", ace_T_label(thermal_frequency), ace_build_time_T1)
@printf("  %s build time:             %.3f s\n", ace_T_label(hot_thermal_frequency), ace_build_time_T3)
println("  formal bath Liouville dim:      $formal_liouville_dimension")
