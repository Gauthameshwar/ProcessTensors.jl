# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: scripts/noisy_quantum_circuit_tester_protocol.jl
# Contributor: Gauthameshwar S.
#
# Draws the SWAP–Z–SWAP tester protocol on the same time grid as the
# noisy-circuit result figure.
#
# Run with:
# julia --project=. scripts/noisy_quantum_circuit_tester_protocol.jl

# Event times match scripts/noisy_quantum_circuit_tester.jl: store, phase, retrieve.
const FINAL_TIME = 5.0
const SWAP_TIMES = (1.0, 3.0)
const Z_TIME = 2.0

import Pkg
plot_env = joinpath(@__DIR__, ".plot_examples_env")
Pkg.activate(plot_env)
if !isfile(joinpath(plot_env, "Manifest.toml"))
    Pkg.add("CairoMakie")
else
    Pkg.instantiate()
end

using CairoMakie
CairoMakie.activate!()

const PT_COLOR = "#3A9A5B"
const PT_FILL = (PT_COLOR, 0.28)
const QUBIT_COLOR = "#E07B2D"
const TESTER_COLOR = "#7A4EAB"
const ARROW_LENGTH_PX = 18.0
const PT_Y = 2.0
const QUBIT_Y = 1.0
const TESTER_Y = 0.0

# Axis window in data units. Pixel scales keep the Z box and the SWAP crosses
# square on the page; one data unit is wider than it is tall.
# A small margin left of t = 0 keeps the initialization circles inside the frame.
# The rails and the tick marks still begin at zero and end at t = 5.
const TIME_Y = -0.62
const X_LIMITS = (-0.07, FINAL_TIME + 0.08)
const Y_LIMITS = (-1.05, 2.55)
const FIGURE_SIZE = (1120, 420)
const PX_PER_X = 1000 / (X_LIMITS[2] - X_LIMITS[1])
const PX_PER_Y = 320 / (Y_LIMITS[2] - Y_LIMITS[1])

function rounded_square(center, y, half_px, radius_px; n=6)
    half_x = half_px / PX_PER_X
    half_y = half_px / PX_PER_Y
    rx = radius_px / PX_PER_X
    ry = radius_px / PX_PER_Y
    x0, x1 = center - half_x, center + half_x
    y0, y1 = y - half_y, y + half_y
    function corner(cx, cy, a0, a1)
        return [Point2f(cx + rx * cos(a), cy + ry * sin(a)) for a in range(a0, a1; length=n)]
    end
    return vcat(
        corner(x1 - rx, y1 - ry, 0, π / 2),
        corner(x0 + rx, y1 - ry, π / 2, π),
        corner(x0 + rx, y0 + ry, π, 3π / 2),
        corner(x1 - rx, y0 + ry, 3π / 2, 2π),
    )
end

arrow_base(tip) = tip - ARROW_LENGTH_PX / PX_PER_X

function arrowhead(tip, y)
    base = arrow_base(tip)
    half_h = 7.0 / PX_PER_Y
    return Point2f[(tip, y), (base, y + half_h), (base, y - half_h)]
end

function cross!(axis, x, y; color=:black, arm_px=11.0, linewidth=4.0)
    dx = arm_px / PX_PER_X
    dy = arm_px / PX_PER_Y
    lines!(axis, [x - dx, x + dx], [y - dy, y + dy]; color=color, linewidth=linewidth)
    lines!(axis, [x - dx, x + dx], [y + dy, y - dy]; color=color, linewidth=linewidth)
    return nothing
end

figure = Figure(size=FIGURE_SIZE, figure_padding=(12, 18, 8, 16))
Label(
    figure[1, 1:2],
    "Noisy qubit with tester interactions";
    fontsize=22,
    font=:bold,
    tellwidth=false,
)

label_axis = Axis(figure[2, 1]; width=78, limits=(0, 1, Y_LIMITS...))
hidedecorations!(label_axis)
hidespines!(label_axis)
text!(label_axis, 1, PT_Y; text="PT", align=(:right, :center), color=PT_COLOR, fontsize=18)
text!(label_axis, 1, QUBIT_Y; text="QUBIT", align=(:right, :center), color=QUBIT_COLOR, fontsize=18)
text!(label_axis, 1, TESTER_Y; text="TESTER", align=(:right, :center), color=TESTER_COLOR, fontsize=18)
text!(label_axis, 1, TIME_Y; text="Time", align=(:right, :center), color=:black, fontsize=18)

axis = Axis(
    figure[2, 2];
    limits=(X_LIMITS..., Y_LIMITS...),
    xgridvisible=false,
    ygridvisible=false,
)
linkyaxes!(label_axis, axis)
colgap!(figure.layout, 1, 6)
hidedecorations!(axis)
hidespines!(axis)
time_axis_end = arrow_base(FINAL_TIME)
lines!(axis, [0.0, time_axis_end], [TIME_Y, TIME_Y]; color=:black, linewidth=1.6)
poly!(axis, arrowhead(FINAL_TIME, TIME_Y); color=:black, strokewidth=0)
tick_drop = 8.0 / PX_PER_Y
for tick in 0:4
    lines!(axis, [tick, tick], [TIME_Y, TIME_Y - tick_drop]; color=:black, linewidth=1.2)
    text!(
        axis,
        tick,
        TIME_Y - tick_drop;
        text=string(tick),
        align=(:center, :top),
        offset=(0, -3),
        fontsize=16,
    )
end

fill_end = arrow_base(FINAL_TIME)
band!(
    axis,
    [0.0, fill_end],
    [QUBIT_Y, QUBIT_Y],
    [PT_Y, PT_Y];
    color=PT_FILL,
)

for (y, color, initialize) in (
        (PT_Y, PT_COLOR, false),
        (QUBIT_Y, QUBIT_COLOR, true),
        (TESTER_Y, TESTER_COLOR, true),
    )
    lines!(axis, [0.0, fill_end], [y, y]; color=color, linewidth=2.6)
    poly!(axis, arrowhead(FINAL_TIME, y); color=color, strokewidth=0)
    if initialize
        scatter!(
            axis,
            [0.0],
            [y];
            marker=:circle,
            markersize=16,
            color=color,
            strokecolor=color,
            strokewidth=1.5,
        )
    end
end

for swap_time in SWAP_TIMES
    lines!(
        axis,
        [swap_time, swap_time],
        [QUBIT_Y, TESTER_Y];
        color=TESTER_COLOR,
        linewidth=4.0,
    )
    cross!(axis, swap_time, QUBIT_Y; color=TESTER_COLOR)
    cross!(axis, swap_time, TESTER_Y; color=TESTER_COLOR)
end

z_box = rounded_square(Z_TIME, TESTER_Y, 36.0, 8.0)
poly!(axis, z_box; color=:white, strokecolor=TESTER_COLOR, strokewidth=2.2)
text!(
    axis,
    Z_TIME,
    TESTER_Y;
    text="Z",
    align=(:center, :center),
    color=TESTER_COLOR,
    fontsize=22,
)

output_dir = joinpath(@__DIR__, "figures")
mkpath(output_dir)
png_path = joinpath(output_dir, "noisy_quantum_circuit_tester_protocol.png")
pdf_path = joinpath(output_dir, "noisy_quantum_circuit_tester_protocol.pdf")
save(png_path, figure; px_per_unit=2)
save(pdf_path, figure)
println("Saved:")
println("  $png_path")
println("  $pdf_path")
