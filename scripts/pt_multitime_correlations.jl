# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: scripts/pt_multitime_correlations.jl
# Contributor: Gauthameshwar S.
#
# Single-mode bath process tensor: two-time correlator heatmaps via `two_time_correlation_seq`.
#
# Run with:
#   julia --project=. scripts/pt_multitime_correlations.jl

import Pkg

const REPO_ROOT = dirname(@__DIR__)
const _PLOT_ENV = joinpath(@__DIR__, ".plot_examples_env")

function activate_plot_examples_env!()
    mkpath(_PLOT_ENV)
    Pkg.activate(_PLOT_ENV)
    manifest = joinpath(_PLOT_ENV, "Manifest.toml")
    if !isfile(manifest)
        Pkg.develop(Pkg.PackageSpec(path=REPO_ROOT))
        Pkg.add([Pkg.PackageSpec(name="CairoMakie"), Pkg.PackageSpec(name="LaTeXStrings")])
    else
        Pkg.instantiate()
    end
    return nothing
end

activate_plot_examples_env!()

using Printf
using Statistics
using CairoMakie
using LaTeXStrings
using CairoMakie: Fixed
using ITensors
using LinearAlgebra
using ProcessTensors

# ------------------------------------------------------------------------------
# 1. Small script utilities
# ------------------------------------------------------------------------------

const STATUS_WIDTH = 90

function print_section(title::AbstractString)
    println()
    println(title)
    println("-" ^ length(title))
end

function update_status(message::AbstractString)
    print("\r", rpad(message, STATUS_WIDTH))
    flush(stdout)
end

function finish_status(message::AbstractString = "")
    print("\r", " "^STATUS_WIDTH, "\r")
    if !isempty(message)
        println(message)
    end
    flush(stdout)
end

set_theme!(
    Theme(
        fontsize=11,
        Axis=(
            titlefont=:bold,
            xlabelfont=:regular,
            ylabelfont=:regular,
        ),
    ),
)

# ------------------------------------------------------------------------------
# 2. Script parameters
# ------------------------------------------------------------------------------

const dt = 0.5
const tf = 5.5
const rho_label = "Dn"
const times = collect(0.0:dt:tf)
const n_times = length(times)

output_dir = joinpath(@__DIR__, "figures")
mkpath(output_dir)

# ------------------------------------------------------------------------------
# 3. Define the physical problem
# ------------------------------------------------------------------------------

print_section("Problem setup")

println("Single-mode process tensor: full (t₁, t₂) grid of two-time correlators.")
println()

@printf("time step dt           = %.4f\n", dt)
@printf("final time t_f         = %.4f\n", tf)
@printf("time points            = %d\n", n_times)
@printf("initial system state   = %s\n", rho_label)

sys_phys = siteinds("S=1/2", 1)
env_phys = siteinds("S=1/2", 1)
env_liouv = liouv_sites(env_phys)

H_sys = OpSum()
H_sys += 1.0, "Sx", 1
system = spin_system(sys_phys, H_sys)

rho_env0_h = to_dm(MPS(env_phys, ["Up"]))
rho_env0_l = to_liouville(rho_env0_h; sites=env_liouv)
H_env = OpSum()
H_env += 1.0, "Sx", 1
coupling = OpSum() + (1.0, "Sz", 1, "Sz", 2)
mode = spin_mode(env_liouv, H_env, rho_env0_l; coupling=coupling)
bath = spin_bath([mode])

rho_sys0_h = to_dm(MPS(sys_phys, [rho_label]))

O_Sz = OpSum()
O_Sz += 2.0, "Sz", 1
O_Sx = OpSum()
O_Sx += 2.0, "Sx", 1
O_Sy = OpSum()
O_Sy += 2.0, "Sy", 1

cases = [
    (L"\langle \sigma_z(t_2) \sigma_z(t_1) \rangle", O_Sz, O_Sz),
    (L"\langle \sigma_z(t_2) \sigma_x(t_1) \rangle", O_Sz, O_Sx),
    (L"\langle \sigma_x(t_2) \sigma_y(t_1) \rangle", O_Sx, O_Sy),
]

# ------------------------------------------------------------------------------
# 4. Main computation
# ------------------------------------------------------------------------------

print_section("Main computation")

grids = Vector{Matrix{ComplexF64}}(undef, length(cases))
titles = Vector{LaTeXString}(undef, length(cases))

for (i, (title, O_A, O_B)) in enumerate(cases)
    println("Case $i/$(length(cases)): $title")

    pt_nsteps = n_times + 1
    pt = build_process_tensor(
        system,
        system.sites[1];
        environment=bath,
        dt=dt,
        nsteps=pt_nsteps,
    )

    grid = fill(NaN + 0im * NaN, n_times, n_times)
    n_pairs = n_times * n_times
    k = 0

    for n1 in 0:(n_times - 1)
        for n2 in 0:(n_times - 1)
            k += 1
            seq = two_time_correlation_seq(
                pt,
                (O_A, n2),
                (O_B, n1);
                rho0=rho_sys0_h,
            )
            grid[n1 + 1, n2 + 1] = evaluate_process(pt, seq)
            if k == 1 || k == n_pairs || k % max(1, n_pairs ÷ 20) == 0
                update_status(@sprintf("  case %d/%d  pair %4d / %4d  t₁=%.2f t₂=%.2f", i, length(cases), k, n_pairs, n1 * dt, n2 * dt))
            end
        end
        finish_status(@sprintf("  finished row t₁ = %.2f (%d/%d)", n1 * dt, n1 + 1, n_times))
    end

    grids[i] = grid
    titles[i] = title
end

# ------------------------------------------------------------------------------
# 5. Diagnostics and sanity checks
# ------------------------------------------------------------------------------

print_section("Diagnostics")

finite_fraction = mean(isfinite.(real.(vcat(grids...))))
@printf("finite correlator fraction = %.3f\n", finite_fraction)
@printf("max |Re correlator|        = %.6f\n", maximum(abs.(real.(vcat(grids...))); init=0.0))
@printf("max |Im correlator|        = %.6f\n", maximum(abs.(imag.(vcat(grids...))); init=0.0))

@assert finite_fraction > 0.99 "Too many non-finite correlator values."

# ------------------------------------------------------------------------------
# 6. Plotting and saved outputs
# ------------------------------------------------------------------------------

print_section("Plotting")

n = length(grids)
t_lo, t_hi = times[1], times[end]
absmax_re = maximum(maximum(abs, view(real(C), isfinite.(real(C)))) for C in grids; init=0.0)
absmax_im = maximum(maximum(abs, view(imag(C), isfinite.(imag(C)))) for C in grids; init=0.0)
absmax_re > 0 || (absmax_re = 1.0)
absmax_im > 0 || (absmax_im = 1.0)

panel = 250
cb_w = 24.0
ncols = n + 1
nrows = 2
fig = Figure(size=(n * panel + cb_w + 80, nrows * panel + 140), figure_padding=2)
fig.layout.alignmode = Outside(12)

panels = fig[2, 1] = GridLayout()
panels.alignmode = Outside(2)
colgap!(panels, 4)
rowgap!(panels, 8)

cmap = :balance
row_specs = [
    (real.(grids), absmax_re, L"\mathrm{Re}\,\langle A(t_2) B(t_1) \rangle"),
    (imag.(grids), absmax_im, L"\mathrm{Im}\,\langle A(t_2) B(t_1) \rangle"),
]
for (row, (grid_views, absmax, cb_label)) in enumerate(row_specs)
    for col in 1:n
        ax = Axis(
            panels[row, col];
            xlabel=row == nrows ? L"t_1" : "",
            ylabel=col == 1 ? L"t_2" : "",
            title=row == 1 ? titles[col] : "",
            aspect=DataAspect(),
            limits=(t_lo, t_hi, t_lo, t_hi),
            xticks=times,
            yticks=times,
        )
        heatmap!(
            ax,
            times,
            times,
            grid_views[col];
            colormap=cmap,
            colorrange=(-absmax, absmax),
            nan_color=(:white, 0.0),
        )
    end
    Colorbar(
        panels[row, ncols];
        colormap=cmap,
        limits=(-absmax, absmax),
        label=cb_label,
        height=panel,
        width=cb_w,
    )
    rowsize!(panels, row, Fixed(panel))
end
for col in 1:n
    colsize!(panels, col, Fixed(panel))
end
colsize!(panels, ncols, Fixed(cb_w))

Label(
    fig[1, 1],
    latexstring(
        L"\mathrm{PT\ single-mode\ two-time\ correlations},\quad \Delta t = ",
        dt,
        L",\quad t_f = ",
        t_hi,
    );
    fontsize=13,
)

rowgap!(fig.layout, 2)
colgap!(fig.layout, 2)
resize_to_layout!(fig)

fig_path = joinpath(output_dir, "pt_multitime_correlations.png")
save(fig_path, fig; px_per_unit=2)

println("Saved figure:")
println("  $fig_path")

# ------------------------------------------------------------------------------
# 7. Final summary
# ------------------------------------------------------------------------------

print_section("Summary")

println("Completed two-time correlation heatmap example.")
println()
println("Main outputs:")
println("  figure: $fig_path")
println()
println("Main diagnostics:")
@printf("  finite correlator fraction = %.3f\n", finite_fraction)
@printf("  correlator cases           = %d\n", length(cases))
