# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: scripts/boundary_driven_xxz_transport.jl
# Contributor: Gauthameshwar S.
#
# Evolves a boundary-driven XXZ spin chain for several anisotropies with
# Liouville-space TDVP and plots current buildup, magnetisation, and bond currents.
#
# Run with:
#   julia --project=. scripts/boundary_driven_xxz_transport.jl

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

using Printf
using Statistics: mean
using CairoMakie
using ITensors
using LaTeXStrings
using ProcessTensors

CairoMakie.activate!()

# ------------------------------------------------------------------------------
# 1. Small script utilities
# ------------------------------------------------------------------------------

const STATUS_WIDTH = 96

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
    if !isempty(message)
        println(message)
    end
    flush(stdout)
end

function xxz_hamiltonian(N::Int, J::Float64, Δ::Float64)
    H = OpSum()
    for j in 1:(N - 1)
        H += J, "Sx", j, "Sx", j + 1
        H += J, "Sy", j, "Sy", j + 1
        H += J * Δ, "Sz", j, "Sz", j + 1
    end
    return H
end

function boundary_jump_ops(N::Int, Γ::Float64, μ::Float64)
    return [
        (Γ * (1 + μ), "S+", 1),
        (Γ * (1 - μ), "S-", 1),
        (Γ * (1 - μ), "S+", N),
        (Γ * (1 + μ), "S-", N),
    ]
end

function magnetisation_liouville_ops(physical_sites, liouville_sites)
    N = length(physical_sites)
    return [
        let observable = OpSum()
            observable += 1.0, "Sz", j
            to_liouville(MPO(observable, physical_sites); sites=liouville_sites)
        end for j in 1:N
    ]
end

function current_liouville_ops(physical_sites, liouville_sites, J::Float64)
    N = length(physical_sites)
    return [
        let observable = OpSum()
            # Continuity current for the XXZ convention used above.
            # Transport-regime labels (ballistic / diffusive / insulating)
            # require system-size scaling and are intentionally not attempted here.
            observable += J, "Sx", j, "Sy", j + 1
            observable += -J, "Sy", j, "Sx", j + 1
            to_liouville(MPO(observable, physical_sites); sites=liouville_sites)
        end for j in 1:(N - 1)
    ]
end

# ------------------------------------------------------------------------------
# 2. User-adjustable parameters
# ------------------------------------------------------------------------------

const N = 8
const exchange = 1.0
const anisotropies = [0.0, 0.5, 1.0]
const bath_coupling = 1.0
const bath_bias = 0.4

const dt = 0.05
const final_time = 12.0
const nsite = 2
const maxdim = 200
const cutoff = 1e-10

const late_time_window = 20
const current_drift_warn = 5e-4
# At N=8 and final_time=12 the mean bond current settles in time, but the
# bond-current profile typically remains sloped. Raise final_time when a flatter
# profile is needed; the script warns instead of claiming a NESS.
const current_spread_warn = 5e-3
const trace_assert_tol = 1e-2
const trace_warn_tol = 5e-4

output_dir = joinpath(@__DIR__, "figures")
mkpath(output_dir)
fig_path = joinpath(output_dir, "boundary_driven_xxz_transport.png")

# ------------------------------------------------------------------------------
# 3. Physical model and observable construction
# ------------------------------------------------------------------------------

print_section("Problem setup")

println("Boundary-driven XXZ transport: Liouville TDVP for several anisotropies.")
println()
@printf("chain length N        = %d\n", N)
@printf("exchange J            = %.3f\n", exchange)
@printf("anisotropies Δ        = %s\n", join(string.(anisotropies), ", "))
@printf("bath coupling Γ       = %.3f\n", bath_coupling)
@printf("boundary bias μ       = %.3f\n", bath_bias)
@printf("timestep dt           = %.3f\n", dt)
@printf("final time            = %.3f\n", final_time)
@printf("TDVP nsite            = %d\n", nsite)
@printf("max bond dimension    = %d\n", maxdim)
@printf("SVD cutoff            = %.1e\n", cutoff)

physical_sites = siteinds("S=1/2", N)
liouville_sites = liouv_sites(physical_sites)
middle_bond = N ÷ 2

jump_ops = boundary_jump_ops(N, bath_coupling, bath_bias)
magnetisation_ops = magnetisation_liouville_ops(physical_sites, liouville_sites)
current_ops = current_liouville_ops(physical_sites, liouville_sites, exchange)

initial_state = MPS(physical_sites, fill("Dn", N))
initial_density = to_dm(initial_state)
initial_density_liouville =
    to_liouville(initial_density; sites=liouville_sites)

nsteps = round(Int, final_time / dt)
@assert isapprox(nsteps * dt, final_time; atol=100eps(Float64))
times = collect(range(0.0; step=dt, length=nsteps + 1))

@assert isapprox(real(tr(initial_density)), 1.0; atol=1e-12)
@assert length(jump_ops) == 4
@assert length(current_ops) == N - 1
@assert 1 <= middle_bond <= N - 1

# ------------------------------------------------------------------------------
# 4. Main computation
# ------------------------------------------------------------------------------

print_section("Main computation")

# Store one NamedTuple per anisotropy. The TDVP call stays visible in the loop.
trajectories = NamedTuple[]

for (run_idx, Δ) in enumerate(anisotropies)
    hamiltonian = xxz_hamiltonian(N, exchange, Δ)
    liouvillian = liouvillian_mpo(
        hamiltonian,
        liouville_sites;
        jump_ops=jump_ops,
    )

    density = copy(initial_density_liouville)
    mean_current = Float64[]
    middle_current = Float64[]
    trace_errors = Float64[]
    bond_dimensions = Int[]

    elapsed = @elapsed begin
        for step in eachindex(times)
            density_trace = tr(to_hilbert(density))
            bond_currents = [
                real(inner(observable, density) / density_trace)
                for observable in current_ops
            ]
            push!(mean_current, mean(bond_currents))
            push!(middle_current, bond_currents[middle_bond])
            push!(trace_errors, abs(density_trace - 1))
            push!(bond_dimensions, maxlinkdim(density))

            if step % 10 == 0 || step == length(times)
                update_status(
                    @sprintf(
                        "  Δ=%.1f  run %d/%d  step %d/%d  t=%.2f  J̄=%.4f  χ=%d",
                        Δ,
                        run_idx,
                        length(anisotropies),
                        step,
                        length(times),
                        times[step],
                        mean_current[end],
                        bond_dimensions[end],
                    )
                )
            end

            step == length(times) && continue

            density = tdvp(
                liouvillian,
                dt,
                density;
                time_step=dt,
                nsite=nsite,
                maxdim=maxdim,
                cutoff=cutoff,
                outputlevel=0,
            )
        end
    end

    density_trace = tr(to_hilbert(density))
    magnetisation_profile = [
        real(inner(observable, density) / density_trace)
        for observable in magnetisation_ops
    ]
    current_profile = [
        real(inner(observable, density) / density_trace)
        for observable in current_ops
    ]

    finish_status(
        @sprintf(
            "  Δ=%.1f complete in %.1f s  (final J̄=%.5f, χmax=%d)",
            Δ,
            elapsed,
            mean_current[end],
            maximum(bond_dimensions),
        )
    )

    push!(
        trajectories,
        (
            anisotropy=Δ,
            times=times,
            mean_current=mean_current,
            middle_current=middle_current,
            magnetisation_profile=magnetisation_profile,
            current_profile=current_profile,
            trace_errors=trace_errors,
            bond_dimensions=bond_dimensions,
            elapsed=elapsed,
        ),
    )
end

# ------------------------------------------------------------------------------
# 5. Diagnostics and sanity checks
# ------------------------------------------------------------------------------

print_section("Diagnostics")

diagnostics = NamedTuple[]

for traj in trajectories
    Δ = traj.anisotropy
    max_trace_error = maximum(traj.trace_errors)
    max_bond_dimension = maximum(traj.bond_dimensions)
    current_spread = maximum(abs.(traj.current_profile .- mean(traj.current_profile)))
    tail_start = max(1, length(traj.mean_current) - late_time_window + 1)
    late_segment = traj.mean_current[tail_start:end]
    late_time_drift = length(late_segment) < 2 ? 0.0 : maximum(abs.(diff(late_segment)))
    magnetisation_drop =
        first(traj.magnetisation_profile) - last(traj.magnetisation_profile)

    println()
    @printf("Δ = %.1f\n", Δ)
    @printf("  runtime                          = %.3f s\n", traj.elapsed)
    @printf("  final mean bond current          = %.6f\n", traj.mean_current[end])
    @printf("  final middle-bond current        = %.6f\n", traj.middle_current[end])
    @printf("  left-to-right magnetisation drop = %.6f\n", magnetisation_drop)
    @printf("  final current spread             = %.3e\n", current_spread)
    @printf("  late-time mean-current drift     = %.3e\n", late_time_drift)
    @printf("  max trace error                  = %.3e\n", max_trace_error)
    @printf("  max bond dimension               = %d\n", max_bond_dimension)

    @assert all(isfinite, traj.mean_current)
    @assert all(isfinite, traj.middle_current)
    @assert all(isfinite, traj.magnetisation_profile)
    @assert all(isfinite, traj.current_profile)
    @assert all(value -> -0.5 - 1e-8 ≤ value ≤ 0.5 + 1e-8, traj.magnetisation_profile)
    @assert max_trace_error < trace_assert_tol

    if max_trace_error > trace_warn_tol
        @warn "Trace drift exceeds the soft script tolerance." Δ max_trace_error trace_warn_tol
    end
    if late_time_drift > current_drift_warn
        @warn(
            "Mean bond current has not settled over the late-time window; the finite chain may still be approaching a nonequilibrium stationary state.",
            Δ,
            late_time_drift,
            current_drift_warn,
            final_time,
        )
    end
    if current_spread > current_spread_warn
        @warn(
            "Final bond-current profile is not spatially uniform enough to claim a settled transport profile.",
            Δ,
            current_spread,
            current_spread_warn,
        )
    end

    push!(
        diagnostics,
        (
            anisotropy=Δ,
            elapsed=traj.elapsed,
            final_mean_current=traj.mean_current[end],
            final_middle_current=traj.middle_current[end],
            magnetisation_drop=magnetisation_drop,
            current_spread=current_spread,
            late_time_drift=late_time_drift,
            max_trace_error=max_trace_error,
            max_bond_dimension=max_bond_dimension,
        ),
    )
end

# ------------------------------------------------------------------------------
# 6. Plotting and saved outputs
# ------------------------------------------------------------------------------

print_section("Plotting")

# Keep one colour / marker / line identity per anisotropy across all panels.
Δ_styles = Dict(
    0.0 => (color=:dodgerblue, linestyle=:solid, marker=:circle),
    0.5 => (color=:darkorange, linestyle=:dash, marker=:rect),
    1.0 => (color=:seagreen, linestyle=:dot, marker=:utriangle),
)

fig = Figure(size=(980, 980))
ga = fig[1, 1] = GridLayout()

ax_current = Axis(
    ga[1, 1];
    xlabel=L"$t$",
    ylabel=L"$\overline{\mathcal{J}}(t)$",
    title="Mean bond current",
)
ax_mag = Axis(
    ga[2, 1];
    xlabel=L"site $j$",
    ylabel=L"$\langle S_j^z \rangle$",
    title="Final magnetisation profile",
)
ax_jprof = Axis(
    ga[3, 1];
    xlabel=L"bond $j$",
    ylabel=L"$\mathcal{J}_j$",
    title="Final bond-current profile",
)

# Isolated-bath targets ±μ/2 as subtle guides (not the interacting NESS values).
hlines!(
    ax_mag,
    [bath_bias / 2, -bath_bias / 2];
    color=(:gray, 0.55),
    linestyle=:dash,
    linewidth=1.6,
)

legend_plots = AbstractPlot[]
legend_labels = Any[]

for traj in trajectories
    Δ = traj.anisotropy
    style = Δ_styles[Δ]
    label = LaTeXString("\\Delta = $(string(Δ))")

    p = lines!(
        ax_current,
        traj.times,
        traj.mean_current;
        color=style.color,
        linestyle=style.linestyle,
        linewidth=2.4,
    )
    scatterlines!(
        ax_mag,
        1:N,
        traj.magnetisation_profile;
        color=style.color,
        linestyle=style.linestyle,
        marker=style.marker,
        linewidth=2.0,
        markersize=11,
    )
    scatterlines!(
        ax_jprof,
        1:(N - 1),
        traj.current_profile;
        color=style.color,
        linestyle=style.linestyle,
        marker=style.marker,
        linewidth=2.0,
        markersize=11,
    )
    push!(legend_plots, p)
    push!(legend_labels, label)
end

axislegend(ax_current, legend_plots, legend_labels; position=:rb, fontsize=11)
axislegend(ax_mag, legend_plots, legend_labels; position=:rt, fontsize=11)
axislegend(ax_jprof, legend_plots, legend_labels; position=:rt, fontsize=11)

xlims!(ax_current, 0, final_time)
xlims!(ax_mag, 0.5, N + 0.5)
xlims!(ax_jprof, 0.5, (N - 1) + 0.5)

Label(
    ga[0, 1],
    "Boundary-driven XXZ transport (N=$N, J=$exchange, Γ=$bath_coupling, μ=$bath_bias)",
    fontsize=16,
    tellwidth=false,
)

rowgap!(ga, 18)
save(fig_path, fig)
println("Saved figure:")
println("  $fig_path")

# ------------------------------------------------------------------------------
# 7. Final summary
# ------------------------------------------------------------------------------

print_section("Summary")

println("Completed boundary-driven XXZ transport script.")
println()
println("Main output:")
println("  figure: $fig_path")
println()
println("Per-anisotropy diagnostics:")
for diag in diagnostics
    @printf(
        "  Δ=%.1f: J̄=%.6f, spread=%.3e, late drift=%.3e, χmax=%d\n",
        diag.anisotropy,
        diag.final_mean_current,
        diag.current_spread,
        diag.late_time_drift,
        diag.max_bond_dimension,
    )
end
println()
@printf("  max trace error over all runs = %.3e\n", maximum(d -> d.max_trace_error, diagnostics))
@printf("  max bond dimension over all runs = %d\n", maximum(d -> d.max_bond_dimension, diagnostics))
println()
println(
    "Interpretation is limited to finite-chain transport under fixed baths: ",
    "anisotropies reshape the magnetisation drop and transmitted current. ",
    "Ballistic / diffusive / insulating classifications require system-size scaling ",
    "and are intentionally not claimed here.",
)
