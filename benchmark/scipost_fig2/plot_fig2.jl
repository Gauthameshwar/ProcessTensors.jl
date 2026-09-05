# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# Read Figure 2 CSV data and render the two-panel paper figure.
#
# Run with:
#   julia --project=. benchmark/scipost_fig2/plot_fig2.jl

import Pkg

const FIGURE_DIR = @__DIR__
const RESULTS_DIR = joinpath(FIGURE_DIR, "results")
const PLOT_ENV = joinpath(dirname(FIGURE_DIR), ".plot_env")

function activate_plot_environment!()
    mkpath(PLOT_ENV)
    Pkg.activate(PLOT_ENV)
    manifest = joinpath(PLOT_ENV, "Manifest.toml")
    if !isfile(manifest)
        Pkg.add([
            Pkg.PackageSpec(name="CairoMakie"),
            Pkg.PackageSpec(name="LaTeXStrings"),
        ])
    else
        Pkg.resolve()
        Pkg.instantiate()
        dependencies = Pkg.project().dependencies
        haskey(dependencies, "LaTeXStrings") || Pkg.add("LaTeXStrings")
    end
    return nothing
end

activate_plot_environment!()

using CairoMakie
using DelimitedFiles
using LaTeXStrings

CairoMakie.activate!()

function load_csv(filename::AbstractString)
    path = joinpath(RESULTS_DIR, filename)
    isfile(path) || error("Missing $path; run run_fig2.jl first.")
    raw, header = readdlm(path, ','; header=true)
    columns = Dict(String(name) => index for (index, name) in enumerate(vec(header)))
    return raw, columns
end

column(raw, columns, name) = raw[:, columns[name]]

function cutoff_label(cutoff)
    exponent = round(Int, log10(cutoff))
    return latexstring("\\epsilon_{\\mathrm{SVD}}=10^{", exponent, "}")
end

function main()
    profiles, profile_columns = load_csv("fig2_profiles.csv")
    dmax_data, dmax_columns = load_csv("fig2_dmax.csv")

    profile_dt = Float64.(column(profiles, profile_columns, "dt"))
    profile_cutoff = Float64.(column(profiles, profile_columns, "cutoff"))
    dmax_dt = Float64.(column(dmax_data, dmax_columns, "dt"))
    dmax_cutoff = Float64.(column(dmax_data, dmax_columns, "cutoff"))
    cutoffs = sort(unique(dmax_cutoff); rev=true)

    environment_path = joinpath(RESULTS_DIR, "environment.txt")
    environment_text = read(environment_path, String)
    match_dt = match(r"dt_reference = ([^\n]+)", environment_text)
    isnothing(match_dt) && error("Missing dt_reference in $environment_path")
    dt_reference = parse(Float64, strip(only(match_dt.captures)))

    colors = [
        RGBf(0.00, 0.45, 0.70),
        RGBf(0.90, 0.55, 0.00),
        RGBf(0.00, 0.62, 0.45),
        RGBf(0.80, 0.20, 0.35),
    ]
    markers = [:circle, :rect, :utriangle, :diamond]
    length(cutoffs) <= length(colors) ||
        error("Plot style supports at most $(length(colors)) cutoffs.")

    figure = Figure(size=(1400, 500), fontsize=26)
    axis_kwargs = (
        xgridvisible=true,
        ygridvisible=true,
        xgridcolor=(:gray, 0.35),
        ygridcolor=(:gray, 0.35),
        xticklabelsize=24,
        yticklabelsize=24,
        xlabelsize=28,
        ylabelsize=28,
        xautolimitmargin=(0.03, 0.03),
        yautolimitmargin=(0.06, 0.08),
    )
    axis_a = Axis(
        figure[1, 1];
        xlabel=L"t_k=k\Delta t",
        ylabel=L"D_k=\dim(\chi_k)",
        yscale=log10,
        axis_kwargs...,
    )
    axis_b = Axis(
        figure[1, 2];
        xlabel=L"\Delta t",
        ylabel=L"D_{\max}",
        axis_kwargs...,
    )

    for (column_index, panel) in ((1, "(a)"), (2, "(b)"))
        Label(
            figure[1, column_index, Left()],
            panel;
            font=:bold,
            fontsize=24,
            tellheight=false,
            tellwidth=false,
            halign=:center,
            valign=:top,
            padding=(0, 58, 0, 0),
        )
    end

    legend_elements = Any[]
    legend_labels = Any[]
    any_capped = false
    for (index, cutoff) in enumerate(cutoffs)
        profile_mask = (profile_dt .== dt_reference) .& (profile_cutoff .== cutoff)
        any(profile_mask) || error("No profile at dt_reference=$dt_reference, cutoff=$cutoff")
        times = Float64.(column(profiles, profile_columns, "t")[profile_mask])
        dimensions = Int.(column(profiles, profile_columns, "D_k")[profile_mask])
        permutation = sortperm(times)
        times = times[permutation]
        dimensions = dimensions[permutation]
        line = lines!(
            axis_a, times, dimensions;
            color=colors[index],
            linestyle=:solid,
            linewidth=2.6,
        )

        dmax_mask = dmax_cutoff .== cutoff
        timesteps = dmax_dt[dmax_mask]
        dmax_values = Int.(column(dmax_data, dmax_columns, "D_max")[dmax_mask])
        capped = lowercase.(string.(column(dmax_data, dmax_columns, "hit_maxdim")[dmax_mask])) .==
                 "true"
        permutation = sortperm(timesteps)
        timesteps = timesteps[permutation]
        dmax_values = dmax_values[permutation]
        capped = capped[permutation]
        lines!(
            axis_b,
            timesteps,
            dmax_values;
            color=(colors[index], 0.55),
            linewidth=2,
        )
        uncapped = .!capped
        scatter!(
            axis_b,
            timesteps[uncapped],
            dmax_values[uncapped];
            marker=markers[index],
            color=colors[index],
            markersize=18,
        )
        if any(capped)
            any_capped = true
            scatter!(
                axis_b,
                timesteps[capped],
                dmax_values[capped];
                marker=markers[index],
                color=:transparent,
                strokecolor=colors[index],
                strokewidth=3,
                markersize=18,
            )
        end

        push!(
            legend_elements,
            [
                LineElement(
                    color=colors[index],
                    linestyle=:solid,
                    linewidth=3,
                ),
                MarkerElement(
                    color=colors[index],
                    marker=markers[index],
                    markersize=14,
                ),
            ],
        )
        push!(legend_labels, cutoff_label(cutoff))
    end

    dmax_values_all = Int.(column(dmax_data, dmax_columns, "D_max"))
    if maximum(dmax_values_all) / minimum(dmax_values_all) >= 10
        axis_b.yscale = log10
    end
    t_max = maximum(Float64.(column(profiles, profile_columns, "t")[profile_dt .== dt_reference]))
    xlims!(axis_a, 0, t_max)
    dt_max = maximum(dmax_dt)
    dt_tick = 0.25
    xlims!(axis_b, 0, dt_max)
    axis_b.xticks = 0:dt_tick:dt_max
    axis_b.xgridvisible = true

    Legend(
        figure[1, 1],
        legend_elements,
        legend_labels;
        tellheight=false,
        tellwidth=false,
        halign=:left,
        valign=:bottom,
        labelsize=16,
        framevisible=true,
        backgroundcolor=(:white, 0.9),
        padding=(8, 8, 8, 8),
        margin=(10, 10, 10, 10),
    )
    if any_capped
        text!(
            axis_b,
            0.98,
            0.04;
            text="open marker: maxdim reached",
            space=:relative,
            align=(:right, :bottom),
            fontsize=15,
        )
    end

    colgap!(figure.layout, 28)
    mkpath(RESULTS_DIR)
    pdf_path = joinpath(RESULTS_DIR, "fig2.pdf")
    png_path = joinpath(RESULTS_DIR, "fig2.png")
    save(pdf_path, figure; pt_per_unit=1)
    save(png_path, figure; px_per_unit=2)
    println("Wrote $pdf_path")
    println("Wrote $png_path")
    return nothing
end

main()
