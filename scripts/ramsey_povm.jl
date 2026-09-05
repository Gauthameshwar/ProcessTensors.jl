# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: scripts/ramsey_povm.jl
# Contributor: Gauthameshwar S.
#
# Builds a three-shot Ramsey protocol with unsharp X POVM-and-reset instruments
# on a thermal bosonic ACE process tensor and plots the protocol and record
# probabilities as two figures.
#
# Run with:
# julia --project=. scripts/ramsey_povm.jl
#
# A matching cache at scripts/.cache/ramsey_povm_pt.jls skips ACE construction.
# Override the path with PT_RAMSEY_CACHE, or force a rebuild with
# PT_RAMSEY_REBUILD=1.

import Pkg

const REPO_ROOT = dirname(@__DIR__)
const PLOT_ENV = joinpath(@__DIR__, ".plot_examples_env")

function activate_plot_environment!()
    mkpath(PLOT_ENV)
    Pkg.activate(PLOT_ENV)
    manifest = joinpath(PLOT_ENV, "Manifest.toml")
    if !isfile(manifest)
        Pkg.develop(Pkg.PackageSpec(path=REPO_ROOT))
        Pkg.add([
            Pkg.PackageSpec(name="CairoMakie"),
            Pkg.PackageSpec(name="LaTeXStrings"),
        ])
    else
        Pkg.instantiate()
    end
    return nothing
end

activate_plot_environment!()

using CairoMakie
using ITensors
using ITensors.Ops: Trotter
using LaTeXStrings
using Logging
using Printf
using Serialization
using ProcessTensors

CairoMakie.activate!()

# -----------------------------------------------------------------------------
# 1. Thermal pure-dephasing process (cached ACE PT)
# -----------------------------------------------------------------------------

# Thermal spin-boson dephasing bath. Alpha is larger than the tester-example
# value so bath memory makes three-shot records deviate from independent
# marginals. Natural units hbar = k_B = 1.
const RAMSEY_CACHE_FORMAT = 1
const RAMSEY_CACHE_KEYS = (
    :N_bath,
    :local_dim,
    :alpha,
    :omega_cutoff,
    :omega_max,
    :thermal_frequency,
    :dt,
    :nsteps,
    :ace_cutoff,
    :ace_maxdim,
)

const N_BATH = 4
const LOCAL_DIM = 3
const ALPHA = 0.20
const OMEGA_C = 4.0
const OMEGA_MAX = 20.0
const TEMPERATURE = 2.5

const DT = 0.15
const NSTEPS = 14
const ACE_CUTOFF = 1e-5
const ACE_MAXDIM = 256

function ramsey_parameters()
    return (;
        N_bath=N_BATH,
        local_dim=LOCAL_DIM,
        alpha=ALPHA,
        omega_cutoff=OMEGA_C,
        omega_max=OMEGA_MAX,
        thermal_frequency=TEMPERATURE,
        dt=DT,
        nsteps=NSTEPS,
        ace_cutoff=ACE_CUTOFF,
        ace_maxdim=ACE_MAXDIM,
        cache_path=get(
            ENV,
            "PT_RAMSEY_CACHE",
            joinpath(@__DIR__, ".cache", "ramsey_povm_pt.jls"),
        ),
    )
end

function ramsey_mode_grid(params)
    frequency_spacing = params.omega_max / params.N_bath
    frequencies = [(k - 0.5) * frequency_spacing for k in 1:params.N_bath]
    spectral_weights =
        2 * params.alpha .* frequencies .* exp.(-frequencies ./ params.omega_cutoff)
    couplings = sqrt.(spectral_weights .* frequency_spacing)
    return frequencies, couplings
end

function ramsey_bath(params)
    frequencies, couplings = ramsey_mode_grid(params)
    bath_sites = siteinds("Boson", params.N_bath; dim=params.local_dim)
    bath_liouville_sites = liouv_sites(bath_sites)
    modes = [
        let
            omega = frequencies[k]
            coupling = couplings[k]
            H_mode = OpSum() + (omega, "N", 1)
            H_coupling = OpSum()
            H_coupling += coupling, "A", 1, "Z", 2
            H_coupling += coupling, "Adag", 1, "Z", 2

            thermal_mode(
                [bath_liouville_sites[k]],
                H_mode,
                params.thermal_frequency;
                coupling=H_coupling,
            )
        end for k in 1:params.N_bath
    ]
    return with_logger(NullLogger()) do
        bosonic_bath(modes)
    end
end

function slim_process_tensor(pt)
    return ProcessTensor(
        pt.core,
        pt.system,
        nothing,
        pt.dt,
        pt.nsteps,
        pt.coupling_site,
    )
end

function build_ramsey_process_tensor(params)
    system_sites = siteinds("Qubit", 1)
    system = qubit_system(system_sites)
    bath = ramsey_bath(params)
    process_tensor = slim_process_tensor(
        build_process_tensor(
            system;
            method=ACE(cutoff=params.ace_cutoff, maxdim=params.ace_maxdim),
            environment=bath,
            dt=params.dt,
            nsteps=params.nsteps,
            sys_alg=Trotter{2}(),
            combine_alg=Trotter{2}(),
            progress=false,
        ),
    )
    metadata = (;
        format=RAMSEY_CACHE_FORMAT,
        N_bath=params.N_bath,
        local_dim=params.local_dim,
        alpha=params.alpha,
        omega_cutoff=params.omega_cutoff,
        omega_max=params.omega_max,
        thermal_frequency=params.thermal_frequency,
        dt=params.dt,
        nsteps=params.nsteps,
        ace_cutoff=params.ace_cutoff,
        ace_maxdim=params.ace_maxdim,
        maxlinkdim=maxlinkdim(process_tensor),
    )
    return (; process_tensor, system_sites, metadata)
end

function save_ramsey_cache(path, payload)
    mkpath(dirname(path))
    open(path, "w") do io
        serialize(io, payload)
    end
    return path
end

function ramsey_cache_mismatch(metadata, params)
    mismatches = String[]
    hasproperty(metadata, :format) || return ["format: missing"]
    metadata.format == RAMSEY_CACHE_FORMAT || return [
        "format: cache=$(metadata.format) script=$RAMSEY_CACHE_FORMAT",
    ]
    for key in RAMSEY_CACHE_KEYS
        cached = getproperty(metadata, key)
        current = getproperty(params, key)
        agrees = cached isa Integer ?
            cached == current :
            isapprox(cached, current; atol=0, rtol=1e-12)
        agrees || push!(mismatches, "$key: cache=$cached script=$current")
    end
    return mismatches
end

function load_or_build_ramsey_process_tensor(params)
    force_rebuild = get(ENV, "PT_RAMSEY_REBUILD", "0") == "1"
    if force_rebuild
        println("PT_RAMSEY_REBUILD=1; constructing a new ACE process tensor.")
    elseif isfile(params.cache_path)
        payload = try
            open(deserialize, params.cache_path)
        catch err
            @warn "Could not read the process-tensor cache; rebuilding." exception = (
                err,
                catch_backtrace(),
            )
            nothing
        end
        if payload !== nothing
            mismatches = ramsey_cache_mismatch(payload.metadata, params)
            if isempty(mismatches)
                println("Found a matching ACE cache; skipping process-tensor construction.")
                println("  cache file:                  $(params.cache_path)")
                @printf("  maximum PT bond dimension:   %d\n", payload.metadata.maxlinkdim)
                return payload
            end
            println("Cached process tensor does not match the script parameters; rebuilding.")
            for line in mismatches
                println("  $line")
            end
        end
    else
        println("No process-tensor cache at $(params.cache_path); building.")
    end

    build_seconds = @elapsed begin
        payload = build_ramsey_process_tensor(params)
    end
    save_ramsey_cache(params.cache_path, payload)
    @printf("  ACE build time:              %.3f s\n", build_seconds)
    @printf("  wrote cache:                 %s\n", params.cache_path)
    @printf("  maximum PT bond dimension:   %d\n", payload.metadata.maxlinkdim)
    return payload
end

params = ramsey_parameters()
payload = load_or_build_ramsey_process_tensor(params)
process_tensor = payload.process_tensor
system_sites = payload.system_sites
rho_reset = to_dm(MPS(system_sites, ["+"]))

# -----------------------------------------------------------------------------
# 2. Unsharp X POVM followed by active reset
# -----------------------------------------------------------------------------

# E_x = (I + x eta X)/2.  The operation inserted into the process tensor is the
# complete measure-and-reprepare map
#
#     A_x(rho) = Tr(E_x rho) rho_reset,
#
# built as a ProductInstrument: the effect on the output leg and the reset
# state on the input leg.
const ETA = 0.90
const OUTCOMES = (-1, 1)

effects = Dict(
    x => OpSum() + (1 / 2, "Id", 1) + (x * ETA / 2, "X", 1)
    for x in OUTCOMES
)
ramsey_instruments = Dict(
    x => observable_measurement(effects[x]) * state_preparation(rho_reset)
    for x in OUTCOMES
)

# -----------------------------------------------------------------------------
# 3. Evaluate every three-shot record
# -----------------------------------------------------------------------------

const READOUT_STEPS = (4, 8, 12)
records = vec(collect(Iterators.product(ntuple(_ -> OUTCOMES, length(READOUT_STEPS))...)))

function branch_probability(record)
    sequence = default_schedule(process_tensor)
    add!(sequence, state_preparation(rho_reset), 0)
    for (step, outcome) in zip(READOUT_STEPS, record)
        add!(sequence, ramsey_instruments[outcome], step)
    end
    add!(sequence, trace_out(), process_tensor.nsteps)
    value = evaluate_process(process_tensor, sequence; progress=false)
    @assert abs(imag(value)) < 1e-7
    return real(value)
end

probabilities = branch_probability.(records)

@assert minimum(probabilities) > -1e-7
@assert abs(sum(probabilities) - 1) < 5e-3

# The product of the three one-shot marginals is the independent-record
# reference. The marginals may differ between rounds, so this comparison does
# not assume stationarity.
marginals = [
    Dict(
        x => sum(
            probabilities[j]
            for j in eachindex(records)
            if records[j][round] == x
        )
        for x in OUTCOMES
    )
    for round in eachindex(READOUT_STEPS)
]

independent_probabilities = [
    prod(marginals[round][record[round]] for round in eachindex(READOUT_STEPS))
    for record in records
]

record_label(record) = join(x == 1 ? "+" : "-" for x in record)
labels = record_label.(records)

@printf("sum of branch probabilities: %.10f\n", sum(probabilities))
println("record     p(PT)       p(marginal)")
for (lab, p, q) in zip(labels, probabilities, independent_probabilities)
    @printf("p(%s)  %10.8f  %10.8f\n", lab, p, q)
end

# -----------------------------------------------------------------------------
# 4. Protocol diagram and record-probability figures
# -----------------------------------------------------------------------------

set_theme!(
    Theme(
        fontsize=20,
        Axis=(
            xgridvisible=false,
            ygridvisible=true,
            topspinevisible=false,
            rightspinevisible=false,
            titlesize=22,
            xlabelsize=22,
            ylabelsize=22,
            xticklabelsize=18,
            yticklabelsize=18,
        ),
        Legend=(labelsize=18,),
    ),
)

const PREPARE_COLOR = "#F7E58B"
const POVM_COLOR = "#FCB06D"
const PT_COLOR = "#3A9A5B"
const PT_FILL = (PT_COLOR, 0.28)
const SYS_Y = 0.0
const PT_Y = 0.28

protocol_figure = Figure(size=(1050, 260), figure_padding=(10, 16, 6, 12))
Label(
    protocol_figure[0, 1],
    "Three repeated Ramsey readouts";
    fontsize=22,
    font=:bold,
    tellwidth=false,
)
Label(
    protocol_figure[1, 1],
    L"\rho_{\mathrm{r}}=|+\rangle\langle+| \qquad E_x^{(\eta)}=\frac{1}{2}(I+x\eta\sigma_x),\ \eta=0.90";
    fontsize=19,
    tellwidth=false,
)

readout_x = (2.50, 5.00, 7.50)
pair_half = 0.14
marker_clearance = 0.10
prepare_x = 0.0
line_start = 0.0
line_end = 10.0
xpad = 1.05

protocol_axis = Axis(
    protocol_figure[2, 1];
    limits=(-xpad, line_end + xpad, -0.48, 0.58),
)
hidedecorations!(protocol_axis)
hidespines!(protocol_axis)

povm_xs = readout_x .- pair_half
reset_xs = readout_x .+ pair_half
evolution_intervals = (
    (line_start, povm_xs[1] - marker_clearance),
    (reset_xs[1] + marker_clearance, povm_xs[2] - marker_clearance),
    (reset_xs[2] + marker_clearance, povm_xs[3] - marker_clearance),
    (reset_xs[3] + marker_clearance, line_end),
)

for (x0, x1) in evolution_intervals
    xs = range(x0, x1; length=24)
    band!(
        protocol_axis,
        xs,
        fill(SYS_Y, length(xs)),
        fill(PT_Y, length(xs));
        color=PT_FILL,
    )
    lines!(protocol_axis, [x0, x1], [SYS_Y, SYS_Y]; color=:black, linewidth=2.4)
end

lines!(protocol_axis, [line_start, line_end], [PT_Y, PT_Y]; color=PT_COLOR, linewidth=2.4)
text!(
    protocol_axis,
    line_start - 0.08,
    PT_Y;
    text="PT",
    align=(:right, :center),
    color=PT_COLOR,
    fontsize=18,
)
text!(
    protocol_axis,
    line_start - 0.08,
    SYS_Y;
    text="QUBIT",
    align=(:right, :center),
    color=:black,
    fontsize=18,
)
scatter!(
    protocol_axis,
    [line_end],
    [PT_Y];
    marker=:rtriangle,
    markersize=18,
    color=PT_COLOR,
)
scatter!(
    protocol_axis,
    [line_end],
    [SYS_Y];
    marker=:rtriangle,
    markersize=18,
    color=:black,
)

scatter!(
    protocol_axis,
    [prepare_x],
    [SYS_Y];
    color=PREPARE_COLOR,
    markersize=18,
    strokecolor=:black,
    strokewidth=2.4,
)
text!(
    protocol_axis,
    prepare_x,
    -0.18;
    text=L"\rho_{\mathrm{r}}",
    align=(:center, :top),
    fontsize=22,
)

outcome_labels = (L"x_1", L"x_2", L"x_3")
for (round, center) in enumerate(readout_x)
    scatter!(
        protocol_axis,
        [povm_xs[round]],
        [SYS_Y];
        color=POVM_COLOR,
        markersize=18,
        strokecolor=:black,
        strokewidth=2.4,
    )
    scatter!(
        protocol_axis,
        [reset_xs[round]],
        [SYS_Y];
        color=PREPARE_COLOR,
        markersize=18,
        strokecolor=:black,
        strokewidth=2.4,
    )
    text!(
        protocol_axis,
        center,
        PT_Y + 0.10;
        text=outcome_labels[round],
        align=(:center, :bottom),
        fontsize=22,
    )
    text!(
        protocol_axis,
        povm_xs[round] - 0.22,
        -0.14;
        text=L"E_x^{(\eta)}",
        align=(:center, :top),
        fontsize=22,
    )
    text!(
        protocol_axis,
        reset_xs[round] + 0.22,
        -0.14;
        text=L"\rho_{\mathrm{r}}",
        align=(:center, :top),
        fontsize=22,
    )
end

record_figure = Figure(size=(1050, 380), figure_padding=(16, 16, 10, 12))
probability_axis = Axis(
    record_figure[1, 1];
    title="Outcome-record probabilities",
    xlabel=L"(x_1 x_2 x_3)",
    ylabel=L"p(x_1,x_2,x_3)",
    xticks=(1:length(labels), labels),
)

positions = collect(1:length(records))
barplot!(
    probability_axis,
    positions .- 0.15,
    probabilities;
    width=0.26,
    color=PT_COLOR,
    label="process-tensor record",
)
barplot!(
    probability_axis,
    positions .+ 0.15,
    independent_probabilities;
    width=0.26,
    color=(POVM_COLOR, 0.92),
    label="product of marginals",
)
axislegend(probability_axis; position=:rt, framevisible=false)
ylims!(probability_axis, 0, 1.12 * maximum(vcat(probabilities, independent_probabilities)))

output_dir = joinpath(@__DIR__, "figures")
mkpath(output_dir)
protocol_pdf = joinpath(output_dir, "ramsey_povm_protocol.pdf")
protocol_png = joinpath(output_dir, "ramsey_povm_protocol.png")
record_pdf = joinpath(output_dir, "ramsey_povm_records.pdf")
record_png = joinpath(output_dir, "ramsey_povm_records.png")

save(protocol_pdf, protocol_figure)
save(protocol_png, protocol_figure; px_per_unit=2)
save(record_pdf, record_figure)
save(record_png, record_figure; px_per_unit=2)

println("Saved:")
println("  $protocol_png")
println("  $record_png")
