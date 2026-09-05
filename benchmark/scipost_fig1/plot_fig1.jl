# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# Read both Figure 1 data files and render the complete 2×2 paper figure.
#
# Run with:
#   julia --project=. benchmark/scipost_fig1/plot_fig1.jl

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
        deps = Pkg.project().dependencies
        haskey(deps, "LaTeXStrings") || Pkg.add("LaTeXStrings")
    end
    return nothing
end

activate_plot_environment!()

using CairoMakie
using DelimitedFiles
using LaTeXStrings
using Printf

CairoMakie.activate!()

function load_csv(filename::AbstractString)
    path = joinpath(RESULTS_DIR, filename)
    isfile(path) || error("Missing $path; run the corresponding data script first.")
    raw, header = readdlm(path, ','; header=true)
    columns = Dict(String(name) => index for (index, name) in enumerate(vec(header)))
    return raw, columns
end

column(raw, columns, name) = raw[:, columns[name]]

function sorted_series(raw, columns, mask; ycolumn::AbstractString)
    times = Float64.(column(raw, columns, "t")[mask])
    values = Float64.(column(raw, columns, ycolumn)[mask])
    permutation = sortperm(times)
    return times[permutation], values[permutation]
end

function positive_series(times, values)
    keep = isfinite.(values) .& (values .> 0)
    return times[keep], values[keep]
end

"""Keep about `n_markers` ED points, always including the endpoints."""
function subsample_markers(times, values; n_markers::Int=22)
    n = length(times)
    n <= n_markers && return times, values
    indices = unique(round.(Int, range(1, n; length=n_markers)))
    return times[indices], values[indices]
end

function main()
    left, left_columns = load_csv("fig1_left.csv")
    right, right_columns = load_csv("fig1_right.csv")

    methods_left = String.(column(left, left_columns, "method"))
    left_dt = Float64.(column(left, left_columns, "dt"))
    timesteps = sort(unique(left_dt); rev=true)
    red_colors = [RGBf(1.00, 0.55, 0.05), RGBf(0.90, 0.12, 0.08), RGBf(0.55, 0.00, 0.08)]
    blue_colors = [RGBf(0.00, 0.78, 0.82), RGBf(0.00, 0.42, 0.72), RGBf(0.00, 0.18, 0.42)]
    n_dt = length(timesteps)
    red_colors = red_colors[1:n_dt]
    blue_colors = blue_colors[1:n_dt]

    figure = Figure(size=(1400, 560), fontsize=26)
    axis_kwargs = (
        xgridvisible=true,
        ygridvisible=true,
        xgridcolor=(:gray, 0.35),
        ygridcolor=(:gray, 0.35),
        xautolimitmargin=(0.0, 0.0),
        yautolimitmargin=(0.0, 0.0),
        xticklabelsize=24,
        yticklabelsize=24,
        xlabelsize=28,
        ylabelsize=28,
        xticksize=8,
        yticksize=8,
    )
    left_grid = figure[1, 1] = GridLayout()
    right_grid = figure[1, 2] = GridLayout()
    axis_a = Axis(
        left_grid[1, 1];
        ylabel=L"\langle \sigma_y(t)\rangle",
        axis_kwargs...,
    )
    axis_b = Axis(
        left_grid[2, 1];
        xlabel=L"t",
        ylabel=L"\varepsilon_{\sigma_y}(t)",
        yscale=log10,
        axis_kwargs...,
    )
    axis_c = Axis(right_grid[1, 1]; axis_kwargs...)
    axis_d = Axis(
        right_grid[2, 1];
        xlabel=L"t",
        yscale=log10,
        axis_kwargs...,
    )
    function panel_letter!(grid, row, letter; right_pad=8, take_width=true)
        return Label(
            grid[row, 1, Left()],
            letter;
            font=:bold,
            fontsize=24,
            tellheight=false,
            tellwidth=take_width,
            valign=:top,
            halign=:center,
            padding=(0, right_pad, 0, 0),
        )
    end
    # Sit on the y-label column: inset from the spine by the tick-label width.
    panel_letter!(left_grid, 1, "(a)"; right_pad=56, take_width=false)
    panel_letter!(left_grid, 2, "(b)"; right_pad=56, take_width=false)
    panel_letter!(right_grid, 1, "(c)")
    panel_letter!(right_grid, 2, "(d)")

    dense_plots = Any[]
    ace_plots = Any[]
    dense_labels = String[]
    ace_labels = String[]
    for (index, dt) in enumerate(timesteps)
        dense_mask = (methods_left .== "exact_pt") .& (left_dt .== dt)
        ace_mask = (methods_left .== "ace") .& (left_dt .== dt)
        dt_text = Printf.@sprintf("%.2f", dt)

        t_dense, sx_dense = sorted_series(left, left_columns, dense_mask; ycolumn="sy")
        dense_line = lines!(
            axis_a, t_dense, sx_dense;
            color=red_colors[index], linewidth=2.4, linestyle=:solid,
        )
        t_ace, sx_ace = sorted_series(left, left_columns, ace_mask; ycolumn="sy")
        ace_line = lines!(
            axis_a, t_ace, sx_ace;
            color=blue_colors[index], linewidth=2.4, linestyle=:dash,
        )
        push!(dense_plots, dense_line)
        push!(ace_plots, ace_line)
        push!(dense_labels, "Dense(): Δt=$dt_text")
        push!(ace_labels, "ACE(): Δt=$dt_text")

        _, error_dense = sorted_series(left, left_columns, dense_mask; ycolumn="eps_sy")
        error_t_dense, error_dense = positive_series(t_dense, error_dense)
        lines!(
            axis_b, error_t_dense, error_dense;
            color=red_colors[index], linewidth=2.4, linestyle=:solid,
        )
        _, error_ace = sorted_series(left, left_columns, ace_mask; ycolumn="eps_sy")
        error_t_ace, error_ace = positive_series(t_ace, error_ace)
        lines!(
            axis_b, error_t_ace, error_ace;
            color=blue_colors[index], linewidth=2.4, linestyle=:dash,
        )
    end

    ed_mask = (methods_left .== "ed") .& (left_dt .== last(timesteps))
    t_ed, sx_ed = sorted_series(left, left_columns, ed_mask; ycolumn="sy")
    t_ed_m, sx_ed_m = subsample_markers(t_ed, sx_ed)
    ed_plot = scatter!(
        axis_a, t_ed_m, sx_ed_m;
        marker=:x, color=:black, markersize=14, strokewidth=0,
    )
    translate!(ed_plot, 0, 0, 10)

    methods_right = String.(column(right, right_columns, "method"))
    sys_orders = Int.(column(right, right_columns, "sys_order"))
    mode_orders = Int.(column(right, right_columns, "mode_order"))
    right_ed_mask = methods_right .== "ed"
    t_right_ed, sx_right_ed = sorted_series(
        right, right_columns, right_ed_mask; ycolumn="sy",
    )

    pair_colors = Dict(
        (1, 1) => RGBf(0.90, 0.45, 0.05),
        (1, 2) => RGBf(0.55, 0.15, 0.65),
        (2, 1) => RGBf(0.10, 0.55, 0.35),
        (2, 2) => RGBf(1.0, 0.84, 0.0),
    )
    pair_styles = Dict(
        (1, 1) => :solid,
        (1, 2) => :dash,
        (2, 1) => :solid,
        (2, 2) => :dash,
    )
    pair_plots = Dict{Tuple{Int,Int},Any}()
    for pair in ((1, 1), (1, 2), (2, 1), (2, 2))
        sys_order, mode_order = pair
        mask = (
            (methods_right .== "ace") .&
            (sys_orders .== sys_order) .&
            (mode_orders .== mode_order)
        )
        times, values = sorted_series(right, right_columns, mask; ycolumn="sy")
        pair_plots[pair] = lines!(
            axis_c, times, values;
            color=pair_colors[pair], linestyle=pair_styles[pair], linewidth=2.6,
        )
        _, errors = sorted_series(right, right_columns, mask; ycolumn="eps_sy")
        error_times, errors = positive_series(times, errors)
        lines!(
            axis_d, error_times, errors;
            color=pair_colors[pair], linestyle=pair_styles[pair], linewidth=2.6,
        )
    end

    t_right_m, sx_right_m = subsample_markers(t_right_ed, sx_right_ed)
    ed_plot_c = scatter!(
        axis_c, t_right_m, sx_right_m;
        marker=:x, color=:black, markersize=14, strokewidth=0,
    )
    translate!(ed_plot_c, 0, 0, 10)

    t_min = minimum(t_ed)
    t_max = maximum(t_ed)
    xlims!(axis_a, t_min, t_max)
    xlims!(axis_b, t_min, t_max)
    xlims!(axis_c, t_min, t_max)
    xlims!(axis_d, t_min, t_max)
    ylims!(axis_a, -1, 1)
    ylims!(axis_c, -1, 1)

    # Makie fills nbanks row-wise, so interleave Dense/ACE to form two columns.
    legend_a_plots = Any[]
    legend_a_labels = String[]
    for i in 1:n_dt
        push!(legend_a_plots, dense_plots[i], ace_plots[i])
        push!(legend_a_labels, dense_labels[i], ace_labels[i])
    end
    push!(legend_a_plots, ed_plot)
    push!(legend_a_labels, "Direct ED")
    Legend(
        left_grid[1, 1],
        legend_a_plots,
        legend_a_labels;
        nbanks=2,
        orientation=:vertical,
        tellheight=false,
        tellwidth=false,
        halign=:left,
        valign=:top,
        labelsize=14,
        framevisible=true,
        backgroundcolor=(:white, 0.88),
        padding=(8, 8, 8, 8),
        rowgap=2,
        colgap=16,
        margin=(10, 10, 10, 10),
    )

    Legend(
        right_grid[1, 1],
        [
            pair_plots[(1, 1)], pair_plots[(2, 1)],
            pair_plots[(1, 2)], pair_plots[(2, 2)],
        ],
        [
            "ACE S1/M1", "ACE S2/M1",
            "ACE S1/M2", "ACE S2/M2",
        ];
        nbanks=2,
        orientation=:vertical,
        tellheight=false,
        tellwidth=false,
        halign=:left,
        valign=:top,
        labelsize=14,
        framevisible=true,
        backgroundcolor=(:white, 0.88),
        padding=(8, 8, 8, 8),
        rowgap=2,
        colgap=16,
        margin=(10, 10, 10, 10),
    )

    linkxaxes!(axis_a, axis_b)
    linkxaxes!(axis_c, axis_d)
    linkyaxes!(axis_a, axis_c)
    hidexdecorations!(axis_a; label=true, ticklabels=true, ticks=true, grid=false, minorticks=true)
    hidexdecorations!(axis_c; label=true, ticklabels=true, ticks=true, grid=false, minorticks=true)
    hideydecorations!(axis_c; label=true, ticklabels=true, ticks=true, grid=false, minorticks=true)
    hideydecorations!(axis_d; label=true, ticklabels=true, ticks=true, grid=false, minorticks=true)
    axis_a.xticklabelspace = 0.0
    axis_c.xticklabelspace = 0.0
    axis_c.yticklabelspace = 0.0
    axis_d.yticklabelspace = 0.0
    rowgap!(left_grid, 24)
    rowgap!(right_grid, 24)
    colgap!(figure.layout, 18)

    mkpath(RESULTS_DIR)
    pdf_path = joinpath(RESULTS_DIR, "fig1.pdf")
    png_path = joinpath(RESULTS_DIR, "fig1.png")
    save(pdf_path, figure; pt_per_unit=1)
    save(png_path, figure; px_per_unit=2)
    println("Wrote $pdf_path")
    println("Wrote $png_path")
    return nothing
end

main()
