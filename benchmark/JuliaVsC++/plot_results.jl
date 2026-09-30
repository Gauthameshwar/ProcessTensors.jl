# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# Plot Julia and C++ ACE construction runtime against maximum PT bond dimension.
#
# Run with:
#   julia --project=benchmark benchmark/JuliaVsC++/plot_results.jl

include(joinpath(@__DIR__, "..", "env.jl"))

const BENCH_DIR = @__DIR__
const RESULTS_DIR = joinpath(BENCH_DIR, "results")

using CairoMakie
using Printf

CairoMakie.activate!()

const COLORS = (:dodgerblue, :darkorange, :seagreen, :purple, :firebrick, :goldenrod)
const MARKERS = (:circle, :rect, :utriangle, :diamond, :pentagon, :hexagon)
const IMPLEMENTATIONS = ("Julia", "C++")

"""Read valid benchmark rows while tolerating a file that is still being written."""
function load_points(
    path;
    implementation,
    model,
    case_column,
    sweep_column,
    case_parser=identity,
)
    if !isfile(path)
        @warn "Benchmark CSV is not available; skipping it" path
        return NamedTuple[]
    end

    lines = readlines(path)
    if isempty(lines)
        @warn "Benchmark CSV is empty; skipping it" path
        return NamedTuple[]
    end

    header = strip.(split(replace(lines[1], '\ufeff' => ""), ','; keepempty=true))
    columns = Dict(name => index for (index, name) in enumerate(header))
    runtime_column = if implementation == "Julia" && haskey(columns, "t_median_s")
        "t_median_s"
    elseif haskey(columns, "elapsed_sec")
        "elapsed_sec"
    else
        ""
    end
    required = ("model", case_column, sweep_column, "maxdim")
    missing_columns = [name for name in required if !haskey(columns, name)]
    isempty(runtime_column) && push!(missing_columns, "elapsed_sec or t_median_s")
    if !isempty(missing_columns)
        @warn "Benchmark CSV lacks required columns; skipping it" path missing_columns
        return NamedTuple[]
    end

    points = NamedTuple[]
    for (row_offset, line) in enumerate(lines[2:end])
        line_number = row_offset + 1
        isempty(strip(line)) && continue
        fields = strip.(split(line, ','; keepempty=true))
        needed = (columns[case_column], columns[sweep_column], columns["maxdim"], columns[runtime_column], columns["model"])
        if length(fields) < maximum(needed)
            @warn "Skipping incomplete benchmark row" path line_number
            continue
        end
        fields[columns["model"]] == model || continue
        try
            runtime = parse(Float64, fields[columns[runtime_column]])
            maxdim = parse(Float64, fields[columns["maxdim"]])
            sweep = parse(Float64, fields[columns[sweep_column]])
            case_key = case_parser(fields[columns[case_column]])
            if !(isfinite(runtime) && runtime > 0 && isfinite(maxdim) && maxdim > 0 && isfinite(sweep))
                @warn "Skipping non-positive or non-finite benchmark row" path line_number
                continue
            end
            push!(points, (; implementation, case_key, sweep, runtime, maxdim))
        catch error
            @warn "Skipping malformed benchmark row" path line_number exception=(error, catch_backtrace())
        end
    end
    return points
end

temperature_key(value) = @sprintf("%.12g", parse(Float64, value))

function ordered_cases(points, preferred)
    available = unique(point.case_key for point in points)
    ordered = [case for case in preferred if case in available]
    append!(ordered, sort(setdiff(available, ordered)))
    return ordered
end

style(index) = (color=COLORS[mod1(index, length(COLORS))],
                marker=MARKERS[mod1(index, length(MARKERS))])

function series!(axis, points, plot_style, implementation)
    isempty(points) && return nothing
    ordered = sort(points; by=point -> point.sweep)
    runtime = getproperty.(ordered, :runtime)
    maxdim = getproperty.(ordered, :maxdim)
    linestyle = implementation == "Julia" ? :solid : :dot
    markercolor = implementation == "Julia" ? plot_style.color : (:white, 0.0)

    if length(ordered) > 1
        lines!(
            axis,
            runtime,
            maxdim;
            color=plot_style.color,
            linestyle,
            linewidth=2,
        )
    end
    scatter!(
        axis,
        runtime,
        maxdim;
        marker=plot_style.marker,
        color=markercolor,
        strokecolor=plot_style.color,
        strokewidth=2,
        markersize=15,
    )
    return nothing
end

const LABEL_ANGLES = ntuple(k -> 2π * (k - 1) / 16, 16)
const LABEL_RADII = (0.022, 0.032, 0.042)

function panel_uv_limits(points)
    xs = getproperty.(points, :runtime)
    ys = getproperty.(points, :maxdim)
    logxmin, logxmax = extrema(log10.(xs))
    ymin, ymax = extrema(ys)
    dlogx = max(logxmax - logxmin, 0.15)
    dy = max(ymax - ymin, 8.0)
    pad_u, pad_v = 0.08 * dlogx, 0.08 * dy
    return (
        logxmin=logxmin - pad_u,
        dlogx=dlogx + 2pad_u,
        ymin=ymin - pad_v,
        dy=dy + 2pad_v,
    )
end

to_uv(x, y, lims) = ((log10(x) - lims.logxmin) / lims.dlogx, (y - lims.ymin) / lims.dy)
from_uv(u, v, lims) = (10^(lims.logxmin + u * lims.dlogx), lims.ymin + v * lims.dy)

function decade_ticks(values)
    lo = floor(Int, log10(minimum(values)))
    hi = ceil(Int, log10(maximum(values)))
    ticks = [exp10(e) for e in lo:hi]
    labels = [rich("10", superscript(string(e))) for e in lo:hi]
    return (ticks, labels)
end

function aabb_overlap(a, b)
    return a.u0 < b.u1 && b.u0 < a.u1 && a.v0 < b.v1 && b.v0 < a.v1
end

function point_in_aabb(u, v, box; pad=0.0)
    return (box.u0 - pad) <= u <= (box.u1 + pad) && (box.v0 - pad) <= v <= (box.v1 + pad)
end

function seg_hits_aabb(u0, v0, u1, v1, box)
    point_in_aabb(u0, v0, box) && return true
    point_in_aabb(u1, v1, box) && return true
    for (su0, sv0, su1, sv1) in (
        (box.u0, box.v0, box.u1, box.v0),
        (box.u1, box.v0, box.u1, box.v1),
        (box.u1, box.v1, box.u0, box.v1),
        (box.u0, box.v1, box.u0, box.v0),
    )
        den = (su1 - su0) * (v1 - v0) - (sv1 - sv0) * (u1 - u0)
        iszero(den) && continue
        t = ((u0 - su0) * (v1 - v0) - (v0 - sv0) * (u1 - u0)) / den
        s = ((u0 - su0) * (sv1 - sv0) - (v0 - sv0) * (su1 - su0)) / den
        0 <= t <= 1 && 0 <= s <= 1 && return true
    end
    return false
end

function label_box(u, v, θ, d; w=0.085, h=0.042)
    cu, cv = u + d * cos(θ), v + d * sin(θ)
    return (u0=cu - w/2, u1=cu + w/2, v0=cv - h/2, v1=cv + h/2, cu=cu, cv=cv)
end

function label_override(point)
    N = round(Int, point.sweep)
    if point.case_key == "unpolarised" && N == 25
        return (θ=π, d=0.034)
    end
    if point.case_key == "unpolarised" && N == 50
        return point.implementation == "Julia" ? (θ=5π/4, d=0.032) : (θ=3π/4, d=0.034)
    end
    if point.implementation == "Julia" && point.case_key == "1" && N == 5
        return (θ=3π/2, d=0.034)
    end
    if point.implementation == "C++" && point.case_key == "1" && N == 5
        return (θ=π/2, d=0.034)
    end
    if point.implementation == "Julia" && point.case_key == "1" && N == 10
        return (θ=11π/6, d=0.05)
    end
    if point.implementation == "Julia" && point.case_key == "1" && N == 20
        return (θ=π, d=0.034)
    end
    return nothing
end

"""Place N=… labels on a ring of radius d around each marker, choosing θ to avoid other markers, polylines, and labels."""
function annotation_poses(points; w=0.085, h=0.042)
    isempty(points) && return NamedTuple[]
    lims = panel_uv_limits(points)
    uv = [to_uv(p.runtime, p.maxdim, lims) for p in points]
    segments = NTuple{4,Float64}[]
    by_series = Dict{Tuple{String,String},Vector{Int}}()
    for (i, p) in enumerate(points)
        push!(get!(Vector{Int}, by_series, (p.implementation, p.case_key)), i)
    end
    for idxs in values(by_series)
        series_order = sort(idxs; by=i -> points[i].sweep)
        for k in 1:(length(series_order) - 1)
            a, b = series_order[k], series_order[k + 1]
            push!(segments, (uv[a][1], uv[a][2], uv[b][1], uv[b][2]))
        end
    end
    n_near = [
        count(j -> j != i && hypot(uv[i][1] - uv[j][1], uv[i][2] - uv[j][2]) < 0.18, eachindex(points))
        for i in eachindex(points)
    ]
    order = sort(eachindex(points); by=i -> -n_near[i])
    placed = NamedTuple[]
    poses = Vector{Union{Nothing,NamedTuple}}(nothing, length(points))
    for i in order
        u, v = uv[i]
        forced = label_override(points[i])
        if forced !== nothing
            box = label_box(u, v, forced.θ, forced.d; w, h)
            push!(placed, box)
            x, y = from_uv(box.cu, box.cv, lims)
            poses[i] = (; x, y)
            continue
        end
        local_u = 0.0
        local_v = 0.0
        nloc = 0
        for (j, (uj, vj)) in enumerate(uv)
            if j != i && hypot(u - uj, v - vj) < 0.22
                local_u += uj
                local_v += vj
                nloc += 1
            end
        end
        nloc > 0 && (local_u /= nloc; local_v /= nloc)
        best = nothing
        best_cost = Inf
        found = false
        for d in LABEL_RADII
            for θ in LABEL_ANGLES
                box = label_box(u, v, θ, d; w, h)
                cost = 25 * d
                (box.u0 < 0 || box.u1 > 1 || box.v0 < 0 || box.v1 > 1) && (cost += 80)
                for (j, (uj, vj)) in enumerate(uv)
                    j == i && continue
                    point_in_aabb(uj, vj, box; pad=0.016) && (cost += 400)
                    cost += 8 / max(hypot(box.cu - uj, box.cv - vj), 0.02)
                end
                for other in placed
                    aabb_overlap(box, other) && (cost += 900)
                    cost += 18 / max(hypot(box.cu - other.cu, box.cv - other.cv), 0.02)
                end
                for (u0, v0, u1, v1) in segments
                    seg_hits_aabb(u0, v0, u1, v1, box) && (cost += 280)
                end
                if nloc > 0
                    cost -= 4 * ((box.cu - u) * (u - local_u) + (box.cv - v) * (v - local_v))
                end
                if cost < best_cost
                    best_cost = cost
                    best = box
                end
                if cost < 8
                    found = true
                    break
                end
            end
            found && break
        end
        box = best === nothing ? label_box(u, v, π/4, LABEL_RADII[end]; w, h) : best
        push!(placed, box)
        x, y = from_uv(box.cu, box.cv, lims)
        poses[i] = (; x, y)
    end
    return [
        (; x=poses[i].x, y=poses[i].y, text="N=$(round(Int, points[i].sweep))", point=points[i])
        for i in eachindex(points)
    ]
end

function annotate_N!(axis, points, color_of)
    for ann in annotation_poses(points)
        text!(
            axis,
            ann.x,
            ann.y;
            text=ann.text,
            color=color_of(ann.point),
            fontsize=14,
            align=(:center, :center),
        )
    end
    return nothing
end

function physical_legend!(position, cases, labels)
    elements = [
        MarkerElement(
            marker=style(index).marker,
            color=style(index).color,
            strokecolor=style(index).color,
            markersize=14,
        )
        for index in eachindex(cases)
    ]
    return Legend(
        position,
        elements,
        labels;
        orientation=:horizontal,
        nbanks=1,
        tellwidth=false,
        labelsize=22,
        framevisible=false,
    )
end

function implementation_legend!(position)
    julia_style = [
        LineElement(color=:gray35, linestyle=:solid, linewidth=2),
        MarkerElement(
            marker=:circle,
            color=:gray35,
            strokecolor=:gray35,
            markersize=14,
        ),
    ]
    cpp_style = [
        LineElement(color=:gray35, linestyle=:dot, linewidth=2),
        MarkerElement(
            marker=:circle,
            color=(:white, 0.0),
            strokecolor=:gray35,
            strokewidth=2,
            markersize=14,
        ),
    ]
    return Legend(
        position,
        [julia_style, cpp_style],
        ["Julia", "C++"];
        orientation=:horizontal,
        nbanks=1,
        tellwidth=false,
        labelsize=22,
        framevisible=false,
    )
end

function main()
    central_points = vcat(
        load_points(
            joinpath(RESULTS_DIR, "julia", "central_spin.csv");
            implementation="Julia",
            model="central_spin",
            case_column="polarisation",
            sweep_column="N",
            case_parser=value -> lowercase(strip(value)),
        ),
        load_points(
            joinpath(RESULTS_DIR, "cpp", "central_spin.csv");
            implementation="C++",
            model="central_spin",
            case_column="polarisation",
            sweep_column="N",
            case_parser=value -> lowercase(strip(value)),
        ),
    )
    spinboson_points = vcat(
        load_points(
            joinpath(RESULTS_DIR, "julia", "spinboson.csv");
            implementation="Julia",
            model="spinboson",
            case_column="kBT_over_Omega",
            sweep_column="N_modes",
            case_parser=temperature_key,
        ),
        load_points(
            joinpath(RESULTS_DIR, "cpp", "spinboson.csv");
            implementation="C++",
            model="spinboson",
            case_column="kBT_over_Omega",
            sweep_column="N_modes",
            case_parser=temperature_key,
        ),
    )

    isempty(central_points) && @warn "No central-spin benchmark points were found"
    isempty(spinboson_points) && @warn "No spin-boson benchmark points were found"

    central_cases = ordered_cases(
        central_points,
        ["polarised", "partial", "unpolarised"],
    )
    spinboson_cases = sort(
        unique(point.case_key for point in spinboson_points);
        by=value -> parse(Float64, value),
    )

    figure = Figure(size=(1500, 720), fontsize=26)
    axis_options = (
        xlabel="runtime (s)",
        ylabel="Dₘₐₓ",
        xscale=log10,
        xgridvisible=true,
        ygridvisible=true,
        xgridcolor=(:gray, 0.3),
        ygridcolor=(:gray, 0.3),
        xticklabelsize=24,
        yticklabelsize=24,
        xlabelsize=28,
        ylabelsize=28,
        xticksize=8,
        yticksize=8,
        titlesize=28,
        titlefont=:bold,
    )
    central_axis = Axis(
        figure[1, 1];
        title="Central-spin process tensor",
        axis_options...,
    )
    spinboson_axis = Axis(
        figure[1, 2];
        title="Spin-boson process tensor",
        axis_options...,
    )
    isempty(central_points) || (central_axis.xticks = decade_ticks(getproperty.(central_points, :runtime)))
    isempty(spinboson_points) || (spinboson_axis.xticks = decade_ticks(getproperty.(spinboson_points, :runtime)))

    Label(
        figure[1, 1, TopLeft()],
        "(a)";
        font=:bold,
        fontsize=28,
        padding=(0, 30, 8, 0),
        halign=:right,
    )
    Label(
        figure[1, 2, TopLeft()],
        "(b)";
        font=:bold,
        fontsize=28,
        padding=(0, 30, 8, 0),
        halign=:right,
    )

    for (case_index, case_key) in enumerate(central_cases)
        plot_style = style(case_index)
        for implementation in IMPLEMENTATIONS
            selected = [
                point for point in central_points
                if point.case_key == case_key && point.implementation == implementation
            ]
            series!(central_axis, selected, plot_style, implementation)
        end
    end
    annotate_N!(central_axis, central_points, p -> style(findfirst(==(p.case_key), central_cases)).color)

    for (case_index, case_key) in enumerate(spinboson_cases)
        plot_style = style(case_index)
        for implementation in IMPLEMENTATIONS
            selected = [
                point for point in spinboson_points
                if point.case_key == case_key && point.implementation == implementation
            ]
            series!(spinboson_axis, selected, plot_style, implementation)
        end
    end
    annotate_N!(spinboson_axis, spinboson_points, p -> style(findfirst(==(p.case_key), spinboson_cases)).color)

    central_legends = GridLayout()
    figure[2, 1] = central_legends
    physical_legend!(
        central_legends[1, 1],
        central_cases,
        ["bath: $(replace(case, "polarised" => "polarized"))" for case in central_cases],
    )
    implementation_legend!(central_legends[2, 1])

    spinboson_legends = GridLayout()
    figure[2, 2] = spinboson_legends
    physical_legend!(
        spinboson_legends[1, 1],
        spinboson_cases,
        ["kBT/Ω = $case" for case in spinboson_cases],
    )
    implementation_legend!(spinboson_legends[2, 1])

    colgap!(figure.layout, 35)
    rowgap!(figure.layout, 8)
    rowsize!(figure.layout, 2, Auto(0.25))

    mkpath(RESULTS_DIR)
    pdf_path = joinpath(RESULTS_DIR, "julia_vs_cpp.pdf")
    png_path = joinpath(RESULTS_DIR, "julia_vs_cpp.png")
    save(pdf_path, figure; pt_per_unit=1)
    save(png_path, figure; px_per_unit=2)
    println("Wrote $pdf_path")
    println("Wrote $png_path")
    return nothing
end

main()
