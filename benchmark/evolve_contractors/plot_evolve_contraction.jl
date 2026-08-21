# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: benchmark/evolve_contractors/plot_evolve_contraction.jl
# Contributor: Gauthameshwar S.
#
# Reads evolve_contraction.csv from benchmark/evolve_contractors/results/ and
# writes a multi-panel comparison of `:evaluate` vs `:closures`.
#
# Run with:
#   julia --project=. benchmark/evolve_contractors/plot_evolve_contraction.jl

import Pkg

const PACK_DIR = @__DIR__
const PLOT_ENV = joinpath(dirname(PACK_DIR), ".plot_env")
const RESULTS_DIR = joinpath(PACK_DIR, "results")
const CSV_NAME = "evolve_contraction.csv"
const FIG_NAME = "evolve_contraction.png"

function activate_plot_env!()
    mkpath(PLOT_ENV)
    Pkg.activate(PLOT_ENV)
    manifest = joinpath(PLOT_ENV, "Manifest.toml")
    if !isfile(manifest)
        Pkg.add(Pkg.PackageSpec(name="CairoMakie"))
    else
        Pkg.resolve()
        Pkg.instantiate()
    end
    return nothing
end

activate_plot_env!()

using CairoMakie
using DelimitedFiles

CairoMakie.activate!()

const COLORS = Dict(
    "evaluate" => :dodgerblue,
    "closures" => :darkorange,
)
const LABELS = Dict(
    "evaluate" => ":evaluate",
    "closures" => ":closures",
)

function load_table(name)
    path = joinpath(RESULTS_DIR, name)
    isfile(path) || error("Missing $path; run benchmark/evolve_contractors/evolve_contraction.jl first.")
    raw, header = readdlm(path, ','; header=true)
    cols = Dict{String,Int}(String(h) => i for (i, h) in enumerate(vec(header)))
    return raw, cols
end

function sorted_xy(raw, cols, contraction, xcol, ycol, mask)
    rows = mask .& (raw[:, cols["contraction"]] .== contraction)
    count(rows) == 0 && return Float64[], Float64[]
    x = Float64.(raw[rows, cols[xcol]])
    y = Float64.(raw[rows, cols[ycol]])
    perm = sortperm(x)
    return x[perm], y[perm]
end

function series!(ax, x, y; color, label, markersize=11, linewidth=2)
    lines!(ax, x, y; color=color, label=label, linewidth=linewidth)
    scatter!(ax, x, y; color=color, markersize=markersize)
    return nothing
end

function main()
    raw, cols = load_table(CSV_NAME)
    nsteps_mask = Int.(raw[:, cols["chi"]]) .== 8
    chi_mask = Int.(raw[:, cols["nsteps"]]) .== 32

    fig = Figure(size=(1180, 720), fontsize=16)
    Label(
        fig[0, 1:3],
        "evolve contraction: :evaluate vs :closures";
        fontsize=20,
        font=:bold,
        tellwidth=false,
    )

    ax_tn = Axis(
        fig[1, 1];
        xlabel="nsteps",
        ylabel="median time (s)",
        xscale=log2,
        yscale=log10,
        title="Time vs nsteps (χ = 8)",
        xticks=[16, 32, 64, 128, 256],
        xminorticksvisible=false,
    )
    ax_mn = Axis(
        fig[1, 2];
        xlabel="nsteps",
        ylabel="allocated memory (MiB)",
        xscale=log2,
        yscale=log10,
        title="Memory vs nsteps (χ = 8)",
        xticks=[16, 32, 64, 128, 256],
        xminorticksvisible=false,
    )
    ax_sn = Axis(
        fig[1, 3];
        xlabel="nsteps",
        ylabel="speedup (:evaluate / :closures)",
        xscale=log2,
        title="Speedup vs nsteps (χ = 8)",
        xticks=[16, 32, 64, 128, 256],
        xminorticksvisible=false,
    )
    ax_tχ = Axis(
        fig[2, 1];
        xlabel="χ",
        ylabel="median time (s)",
        xscale=log2,
        yscale=log10,
        title="Time vs χ (nsteps = 32)",
        xticks=[4, 8, 16, 32, 64, 128, 256],
        xminorticksvisible=false,
    )
    ax_mχ = Axis(
        fig[2, 2];
        xlabel="χ",
        ylabel="allocated memory (MiB)",
        xscale=log2,
        yscale=log10,
        title="Memory vs χ (nsteps = 32)",
        xticks=[4, 8, 16, 32, 64, 128, 256],
        xminorticksvisible=false,
    )
    ax_sχ = Axis(
        fig[2, 3];
        xlabel="χ",
        ylabel="speedup (:evaluate / :closures)",
        xscale=log2,
        title="Speedup vs χ (nsteps = 32)",
        xticks=[4, 8, 16, 32, 64, 128, 256],
        xminorticksvisible=false,
    )

    for contraction in ("evaluate", "closures")
        color = COLORS[contraction]
        label = LABELS[contraction]
        xn, tn = sorted_xy(raw, cols, contraction, "nsteps", "t_median_s", nsteps_mask)
        _, mn = sorted_xy(raw, cols, contraction, "nsteps", "allocated_mib", nsteps_mask)
        xχ, tχ = sorted_xy(raw, cols, contraction, "chi", "t_median_s", chi_mask)
        _, mχ = sorted_xy(raw, cols, contraction, "chi", "allocated_mib", chi_mask)
        series!(ax_tn, xn, tn; color=color, label=label)
        series!(ax_mn, xn, mn; color=color, label=label)
        series!(ax_tχ, xχ, tχ; color=color, label=label)
        series!(ax_mχ, xχ, mχ; color=color, label=label)
    end

    n_eval, t_eval = sorted_xy(raw, cols, "evaluate", "nsteps", "t_median_s", nsteps_mask)
    n_close, t_close = sorted_xy(raw, cols, "closures", "nsteps", "t_median_s", nsteps_mask)
    χ_eval, tχ_eval = sorted_xy(raw, cols, "evaluate", "chi", "t_median_s", chi_mask)
    χ_close, tχ_close = sorted_xy(raw, cols, "closures", "chi", "t_median_s", chi_mask)
    series!(ax_sn, n_eval, t_eval ./ t_close; color=:seagreen, label="time")
    series!(ax_sχ, χ_eval, tχ_eval ./ tχ_close; color=:seagreen, label="time")
    _, m_eval = sorted_xy(raw, cols, "evaluate", "nsteps", "allocated_mib", nsteps_mask)
    _, m_close = sorted_xy(raw, cols, "closures", "nsteps", "allocated_mib", nsteps_mask)
    _, mχ_eval = sorted_xy(raw, cols, "evaluate", "chi", "allocated_mib", chi_mask)
    _, mχ_close = sorted_xy(raw, cols, "closures", "chi", "allocated_mib", chi_mask)
    series!(ax_sn, n_eval, m_eval ./ m_close; color=:purple, label="memory")
    series!(ax_sχ, χ_eval, mχ_eval ./ mχ_close; color=:purple, label="memory")

    axislegend(ax_tn; position=:lt, framevisible=false)
    axislegend(ax_sn; position=:lt, framevisible=false)
    axislegend(ax_sχ; position=:lt, framevisible=false)

    colgap!(fig.layout, 18)
    rowgap!(fig.layout, 16)
    out = joinpath(RESULTS_DIR, FIG_NAME)
    save(out, fig)
    println("Wrote $out")
    return nothing
end

main()
