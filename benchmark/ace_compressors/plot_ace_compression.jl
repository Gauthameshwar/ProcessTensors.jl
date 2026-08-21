# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: benchmark/ace_compressors/plot_ace_compression.jl
# Contributor: Gauthameshwar S.
#
# Reads ACE compression CSVs from benchmark/ace_compressors/results/ and writes
# error-vs-cost and scaling figures next to them.
#
# Run with:
#   julia -t auto --project=. benchmark/ace_compressors/plot_ace_compression.jl

import Pkg

const PACK_DIR = @__DIR__
const PLOT_ENV = joinpath(dirname(PACK_DIR), ".plot_env")
const RESULTS_DIR = joinpath(PACK_DIR, "results")

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

function load_table(name)
    path = joinpath(RESULTS_DIR, name)
    isfile(path) || error("Missing $path; run the corresponding bench script first.")
    raw, header = readdlm(path, ','; header=true)
    cols = Dict{String,Int}(String(h) => i for (i, h) in enumerate(vec(header)))
    return raw, cols
end

function main()
    mkpath(RESULTS_DIR)
    colors = Dict("zipup" => :dodgerblue, "canonzip" => :darkorange)

    bench_path = joinpath(RESULTS_DIR, "ace_compression_benchmark.csv")
    if isfile(bench_path)
        raw, cols = load_table("ace_compression_benchmark.csv")
        fig_t = Figure(size=(720, 480))
        ax_t = Axis(
            fig_t[1, 1];
            xlabel="minimum build time (s)",
            ylabel="process error vs canonzip ε=1e-13",
            yscale=log10,
            title="ACE compression: error vs time",
            xgridvisible=true,
            ygridvisible=true,
        )
        fig_χ = Figure(size=(720, 480))
        ax_χ = Axis(
            fig_χ[1, 1];
            xlabel="χmax",
            ylabel="process error vs canonzip ε=1e-13",
            yscale=log10,
            title="ACE compression: error vs χ",
            xgridvisible=true,
            ygridvisible=true,
        )
        fig_m = Figure(size=(720, 480))
        ax_m = Axis(
            fig_m[1, 1];
            xlabel="allocated memory (MiB)",
            ylabel="process error vs canonzip ε=1e-13",
            yscale=log10,
            title="ACE compression: error vs memory",
            xgridvisible=true,
            ygridvisible=true,
        )
        for strategy in ("zipup", "canonzip")
            mask = raw[:, cols["strategy"]] .== strategy
            count(mask) == 0 && continue
            t = Float64.(raw[mask, cols["t_build"]])
            χ = Float64.(raw[mask, cols["chi_max"]])
            mem = Float64.(raw[mask, cols["allocated_mib"]])
            err = max.(Float64.(raw[mask, cols["eps_rho"]]), 1e-16)
            scatter!(ax_t, t, err; color=colors[strategy], label=strategy, markersize=12)
            lines!(ax_t, t, err; color=colors[strategy])
            scatter!(ax_χ, χ, err; color=colors[strategy], label=strategy, markersize=12)
            lines!(ax_χ, χ, err; color=colors[strategy])
            scatter!(ax_m, mem, err; color=colors[strategy], label=strategy, markersize=12)
            lines!(ax_m, mem, err; color=colors[strategy])
        end
        axislegend(ax_t; position=:rt)
        axislegend(ax_χ; position=:rt)
        axislegend(ax_m; position=:rt)
        save(joinpath(RESULTS_DIR, "ace_compression_error_vs_time.png"), fig_t)
        save(joinpath(RESULTS_DIR, "ace_compression_error_vs_chi.png"), fig_χ)
        save(joinpath(RESULTS_DIR, "ace_compression_error_vs_memory.png"), fig_m)
        println("Wrote ACE compression figures.")
    else
        @warn "Skipping ACE compression figures; missing $bench_path"
    end

    scaling_path = joinpath(RESULTS_DIR, "ace_central_spin_scaling.csv")
    if isfile(scaling_path)
        raw, cols = load_table("ace_central_spin_scaling.csv")
        fig = Figure(size=(720, 960))
        ax_t = Axis(
            fig[1, 1];
            xlabel="N",
            ylabel="minimum build time (s)",
            title="Polarized central-spin ACE scaling",
            xticklabelsvisible=false,
            xlabelvisible=false,
        )
        ax_χ = Axis(
            fig[2, 1];
            xlabel="N",
            ylabel="χmax",
            xticklabelsvisible=false,
            xlabelvisible=false,
        )
        ax_m = Axis(fig[3, 1]; xlabel="N", ylabel="allocated memory (MiB)")
        for strategy in ("zipup", "canonzip")
            mask = raw[:, cols["strategy"]] .== strategy
            count(mask) == 0 && continue
            N = Float64.(raw[mask, cols["N"]])
            t = Float64.(raw[mask, cols["t_build"]])
            χ = Float64.(raw[mask, cols["chi_max"]])
            mem = Float64.(raw[mask, cols["allocated_mib"]])
            perm = sortperm(N)
            lines!(ax_t, N[perm], t[perm]; color=colors[strategy], label=strategy, linewidth=2)
            scatter!(ax_t, N[perm], t[perm]; color=colors[strategy])
            lines!(ax_χ, N[perm], χ[perm]; color=colors[strategy], label=strategy, linewidth=2)
            scatter!(ax_χ, N[perm], χ[perm]; color=colors[strategy])
            lines!(ax_m, N[perm], mem[perm]; color=colors[strategy], label=strategy, linewidth=2)
            scatter!(ax_m, N[perm], mem[perm]; color=colors[strategy])
        end
        axislegend(ax_t; position=:lt)
        rowgap!(fig.layout, 12)
        save(joinpath(RESULTS_DIR, "ace_central_spin_scaling.png"), fig)
        println("Wrote central-spin scaling figure.")
    else
        @warn "Skipping scaling figure; missing $scaling_path"
    end

    return nothing
end

main()
