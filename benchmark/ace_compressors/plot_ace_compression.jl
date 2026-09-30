# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: benchmark/ace_compressors/plot_ace_compression.jl
# Contributor: Gauthameshwar S.
#
# Reads ACE compression CSVs and writes one 2×3 paper figure.
#
# Run with:
#   julia --project=benchmark benchmark/ace_compressors/plot_ace_compression.jl

include(joinpath(@__DIR__, "..", "env.jl"))

const PACK_DIR = @__DIR__
const RESULTS_DIR = joinpath(PACK_DIR, "results")

using CairoMakie
using DelimitedFiles
using LaTeXStrings

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
    colors = Dict(
        "zipup_cpp" => :seagreen,
        "canonzip" => :darkorange,
    )
    strategies = ("zipup_cpp", "canonzip")
    comparison_cutoffs = (1e-6, 1e-8, 1e-10, 1e-12)
    cutoff_markers = Dict(
        1e-6 => :circle,
        1e-8 => :rect,
        1e-10 => :utriangle,
        1e-12 => :diamond,
    )
    cutoff_labels = Dict(
        1e-6 => L"\varepsilon=10^{-6}",
        1e-8 => L"\varepsilon=10^{-8}",
        1e-10 => L"\varepsilon=10^{-10}",
        1e-12 => L"\varepsilon=10^{-12}",
    )
    scaling, scaling_cols = load_table("ace_central_spin_scaling.csv")
    bench, bench_cols = load_table("ace_compression_benchmark.csv")
    threads, thread_cols = load_table("ace_thread_scaling.csv")

    figure = Figure(size=(1600, 760), fontsize=26)
    axis_kwargs = (
        xgridvisible=true,
        ygridvisible=true,
        xgridcolor=(:gray, 0.35),
        ygridcolor=(:gray, 0.35),
        xticklabelsize=24,
        yticklabelsize=24,
        xlabelsize=28,
        ylabelsize=28,
        xticksize=8,
        yticksize=8,
        ylabelpadding=8,
        titlefont=:bold,
        titlesize=28,
        titlealign=:left,
        titlegap=4,
    )
    n_ticks = [1, 5, 10, 20, 30, 40, 50, 70]
    axis_a = Axis(
        figure[1, 1];
        title="(a)",
        xlabel=L"N",
        ylabel="build time (s)",
        xticks=n_ticks,
        axis_kwargs...,
    )
    axis_b = Axis(
        figure[1, 2];
        title="(b)",
        xlabel=L"N",
        ylabel="memory (MiB)",
        xticks=n_ticks,
        axis_kwargs...,
    )
    axis_c = Axis(
        figure[1, 3];
        title="(c)",
        xlabel=L"N",
        ylabel=L"\chi_{\max}",
        xticks=n_ticks,
        axis_kwargs...,
    )
    axis_d = Axis(
        figure[2, 1];
        title="(d)",
        xlabel="build time (s)",
        ylabel=L"\max_k\|\rho_k-\rho_k^{\mathrm{ref}}\|_F",
        yscale=log10,
        axis_kwargs...,
    )
    axis_e = Axis(
        figure[2, 2];
        title="(e)",
        xlabel="memory (MiB)",
        ylabel=L"\max_k\|\rho_k-\rho_k^{\mathrm{ref}}\|_F",
        yscale=log10,
        axis_kwargs...,
    )
    axis_f = Axis(
        figure[2, 3];
        title="(f)",
        xlabel="BLAS threads",
        ylabel="build time (s)",
        xscale=log10,
        yscale=log10,
        xticks=[1, 2, 4, 8, 16, 32],
        axis_kwargs...,
    )

    for strategy in strategies
        mask = scaling[:, scaling_cols["strategy"]] .== strategy
        count(mask) == 0 && continue
        N = Float64.(scaling[mask, scaling_cols["N"]])
        t = Float64.(scaling[mask, scaling_cols["t_median_s"]])
        χ = Float64.(scaling[mask, scaling_cols["chi_max"]])
        mem = Float64.(scaling[mask, scaling_cols["allocated_mib"]])
        perm = sortperm(N)
        lines!(axis_a, N[perm], t[perm]; color=colors[strategy], label=strategy, linewidth=2)
        scatter!(axis_a, N[perm], t[perm]; color=colors[strategy], markersize=11)
        lines!(axis_b, N[perm], mem[perm]; color=colors[strategy], linewidth=2)
        scatter!(axis_b, N[perm], mem[perm]; color=colors[strategy], markersize=11)
        lines!(axis_c, N[perm], χ[perm]; color=colors[strategy], linewidth=2)
        scatter!(axis_c, N[perm], χ[perm]; color=colors[strategy], markersize=11)
    end

    for strategy in strategies
        mask = (bench[:, bench_cols["strategy"]] .== strategy) .&
               in.(Float64.(bench[:, bench_cols["cutoff"]]), Ref(comparison_cutoffs))
        count(mask) == 0 && continue
        t = Float64.(bench[mask, bench_cols["t_median_s"]])
        mem = Float64.(bench[mask, bench_cols["allocated_mib"]])
        err = max.(Float64.(bench[mask, bench_cols["eps_rho"]]), 1e-16)
        cutoffs = Float64.(bench[mask, bench_cols["cutoff"]])
        perm = sortperm(cutoffs; rev=true)
        t, mem, err, cutoffs = t[perm], mem[perm], err[perm], cutoffs[perm]
        lines!(axis_d, t, err; color=colors[strategy])
        lines!(axis_e, mem, err; color=colors[strategy])
        for i in eachindex(cutoffs)
            marker = cutoff_markers[cutoffs[i]]
            scatter!(axis_d, [t[i]], [err[i]]; color=colors[strategy], marker, markersize=14)
            scatter!(axis_e, [mem[i]], [err[i]]; color=colors[strategy], marker, markersize=14)
        end
    end

    style_for_case = Dict("easy" => :solid, "hard" => :dash)
    chi_labels = String[]
    for case_name in ("easy", "hard")
        linestyle = style_for_case[case_name]
        for strategy in strategies
            mask = (threads[:, thread_cols["case"]] .== case_name) .&
                   (threads[:, thread_cols["strategy"]] .== strategy)
            count(mask) == 0 && continue
            n = Float64.(threads[mask, thread_cols["blas_threads"]])
            t = Float64.(threads[mask, thread_cols["t_median_s"]])
            χ = Int.(threads[mask, thread_cols["chi_max"]])
            perm = sortperm(n)
            lines!(
                axis_f,
                n[perm],
                t[perm];
                color=colors[strategy],
                linestyle,
                linewidth=2,
            )
            scatter!(
                axis_f,
                n[perm],
                t[perm];
                color=colors[strategy],
                markersize=11,
            )
            unique_chi = unique(χ)
            label = length(unique_chi) == 1 ?
                "$(case_name) $strategy, χ=$(only(unique_chi))" :
                "$(case_name) $strategy, χ=$(join(unique_chi, '/'))"
            push!(chi_labels, label)
        end
    end

    axislegend(axis_a; position=:lt, labelsize=22, margin=(8, 8, 8, 8), framevisible=true)
    cutoff_legend = [MarkerElement(; color=:gray, marker=cutoff_markers[ε], markersize=14)
                     for ε in comparison_cutoffs]
    cutoff_texts = [cutoff_labels[ε] for ε in comparison_cutoffs]
    for axis in (axis_d, axis_e)
        axislegend(axis, cutoff_legend, cutoff_texts; position=:rt, labelsize=22)
    end
    physics_legend = [
        LineElement(color=:gray35, linestyle=:solid, linewidth=2),
        LineElement(color=:gray35, linestyle=:dash, linewidth=2),
    ]
    axislegend(
        axis_f,
        physics_legend,
        ["spin (easy)", "boson (hard)"];
        position=(0.02, 0.36),
        labelsize=22,
        margin=(8, 8, 8, 8),
        framevisible=true,
    )
    if !isempty(chi_labels)
        println("Panel (f) χ labels: ", join(chi_labels, "; "))
    end
    colgap!(figure.layout, 22)
    rowgap!(figure.layout, 16)

    pdf_path = joinpath(RESULTS_DIR, "ace_compression.pdf")
    png_path = joinpath(RESULTS_DIR, "ace_compression.png")
    save(pdf_path, figure; pt_per_unit=1)
    save(png_path, figure; px_per_unit=2)
    println("Wrote $pdf_path")
    println("Wrote $png_path")
    return nothing
end

main()
