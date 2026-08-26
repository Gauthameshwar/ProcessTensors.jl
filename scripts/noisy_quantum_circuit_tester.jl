# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: scripts/noisy_quantum_circuit_tester.jl
# Contributor: Gauthameshwar S.
#
# Builds or reuses a cached thermal-bosonic ACE process tensor and contracts
# spectator, store–wait–retrieve, phase-tag, and idle-time-sweep tester circuits.
#
# Run with:
# julia --project=. scripts/noisy_quantum_circuit_tester.jl
#
# A matching cache at scripts/.cache/noisy_quantum_circuit_tester_pt.jls skips ACE
# construction. Override the path with PT_TESTER_CACHE, or force a rebuild
# with PT_TESTER_REBUILD=1.

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
        Pkg.resolve()
        Pkg.instantiate()
    end
    return nothing
end

activate_plot_environment!()

using LinearAlgebra
using Logging
using Printf
using Serialization
using CairoMakie
using ITensors
using LaTeXStrings
using ProcessTensors

CairoMakie.activate!()

# -----------------------------------------------------------------------------
# Bath parameters, ACE construction, and on-disk cache
# -----------------------------------------------------------------------------

const TESTER_MEMORY_CACHE_FORMAT = 1
const CACHE_COMPARE_KEYS = (
    :N_bath,
    :local_dim,
    :alpha,
    :omega_cutoff,
    :omega_max,
    :thermal_frequency,
    :dt,
    :final_time,
    :nsteps,
    :ace_cutoff,
    :ace_maxdim,
)

function tester_memory_parameters()
    N_bath = parse(Int, get(ENV, "PT_TESTER_NBATH", "24"))
    local_dim = parse(Int, get(ENV, "PT_TESTER_LOCAL_DIM", "3"))
    alpha = parse(Float64, get(ENV, "PT_TESTER_ALPHA", "0.008"))
    omega_cutoff = 4.0
    omega_max = parse(Float64, get(ENV, "PT_TESTER_OMEGA_MAX", "20.0"))
    thermal_frequency = parse(Float64, get(ENV, "PT_TESTER_THERMAL", "2.5"))
    dt = parse(Float64, get(ENV, "PT_TESTER_DT", "0.20"))
    final_time = parse(Float64, get(ENV, "PT_TESTER_FINAL_TIME", "5.0"))
    nsteps = round(Int, final_time / dt) + 1
    ace_cutoff = parse(Float64, get(ENV, "PT_TESTER_ACE_CUTOFF", "1e-5"))
    ace_maxdim = parse(Int, get(ENV, "PT_TESTER_ACE_MAXDIM", "512"))
    cache_path = get(
        ENV,
        "PT_TESTER_CACHE",
        joinpath(@__DIR__, ".cache", "noisy_quantum_circuit_tester_pt.jls"),
    )
    isapprox((nsteps - 1) * dt, final_time; atol=100eps(Float64)) || throw(
        ArgumentError(
            "tester_memory_parameters: (nsteps-1)*dt must equal final_time; " *
            "got nsteps=$nsteps, dt=$dt, final_time=$final_time.",
        ),
    )
    return (;
        N_bath,
        local_dim,
        alpha,
        omega_cutoff,
        omega_max,
        thermal_frequency,
        dt,
        final_time,
        nsteps,
        ace_cutoff,
        ace_maxdim,
        cache_path,
    )
end

function print_tester_memory_parameters(params)
    @printf("  bath modes:                  %d\n", params.N_bath)
    @printf("  boson local dimension:       %d\n", params.local_dim)
    @printf("  Ohmic strength alpha:        %.4f\n", params.alpha)
    @printf("  cutoff frequency omega_c:    %.3f\n", params.omega_cutoff)
    @printf("  thermal frequency kBT/hbar:  %.3f\n", params.thermal_frequency)
    @printf("  timestep:                    %.4f\n", params.dt)
    @printf("  final time:                  %.3f\n", params.final_time)
    @printf("  nsteps:                      %d\n", params.nsteps)
    @printf("  ACE cutoff:                  %.2e\n", params.ace_cutoff)
    @printf("  ACE maxdim:                  %d\n", params.ace_maxdim)
    return nothing
end

function tester_memory_mode_grid(params)
    frequency_spacing = params.omega_max / params.N_bath
    frequencies = [(k - 0.5) * frequency_spacing for k in 1:params.N_bath]
    spectral_weights =
        2 * params.alpha .* frequencies .* exp.(-frequencies ./ params.omega_cutoff)
    couplings = sqrt.(spectral_weights .* frequency_spacing)
    return frequencies, couplings
end

function tester_memory_bath(params)
    frequencies, couplings = tester_memory_mode_grid(params)
    bath_sites = siteinds("Boson", params.N_bath; dim=params.local_dim)
    bath_liouville_sites = liouv_sites(bath_sites)
    modes = [
        let
            omega = frequencies[k]
            coupling = couplings[k]
            occupations = 0:(params.local_dim - 1)
            thermal_weights =
                exp.(-omega .* occupations ./ params.thermal_frequency)
            thermal_weights ./= sum(thermal_weights)
            number_states = [
                MPS([bath_sites[k]], [string(n)])
                for n in occupations
            ]
            rho_mode = to_liouville(
                to_dm(number_states; coeffs=thermal_weights);
                sites=[bath_liouville_sites[k]],
            )

            H_mode = OpSum() + (omega, "N", 1)
            H_coupling = OpSum()
            H_coupling += coupling, "A", 1, "Z", 2
            H_coupling += coupling, "Adag", 1, "Z", 2

            bosonic_mode(
                [bath_liouville_sites[k]],
                H_mode,
                rho_mode;
                coupling=H_coupling,
            )
        end for k in 1:params.N_bath
    ]
    return Logging.with_logger(Logging.NullLogger()) do
        bosonic_bath(modes)
    end
end

function build_tester_memory_process_tensor(params)
    system_sites = siteinds("Qubit", 1)
    system = qubit_system(system_sites)
    bath = tester_memory_bath(params)
    process_tensor = build_process_tensor(
        system;
        method=ACE(cutoff=params.ace_cutoff, maxdim=params.ace_maxdim),
        environment=bath,
        dt=params.dt,
        nsteps=params.nsteps,
        sys_alg=ITensors.Ops.Trotter{2}(),
        combine_alg=ITensors.Ops.Trotter{2}(),
        progress=true,
    )
    cached = ProcessTensor(
        process_tensor.core,
        process_tensor.system,
        nothing,
        process_tensor.dt,
        process_tensor.nsteps,
        process_tensor.coupling_site,
    )
    metadata = (;
        format=TESTER_MEMORY_CACHE_FORMAT,
        N_bath=params.N_bath,
        local_dim=params.local_dim,
        alpha=params.alpha,
        omega_cutoff=params.omega_cutoff,
        omega_max=params.omega_max,
        thermal_frequency=params.thermal_frequency,
        dt=params.dt,
        final_time=params.final_time,
        nsteps=params.nsteps,
        ace_cutoff=params.ace_cutoff,
        ace_maxdim=params.ace_maxdim,
        maxlinkdim=maxlinkdim(cached),
    )
    return (;
        process_tensor=cached,
        system_sites,
        metadata,
    )
end

function save_tester_memory_cache(path, payload)
    mkpath(dirname(path))
    open(path, "w") do io
        serialize(io, payload)
    end
    return path
end

function cache_parameter_mismatch(metadata, params)
    mismatches = String[]
    hasproperty(metadata, :format) || return ["format: missing"]
    metadata.format == TESTER_MEMORY_CACHE_FORMAT || return [
        "format: cache=$(metadata.format) script=$TESTER_MEMORY_CACHE_FORMAT",
    ]
    for key in CACHE_COMPARE_KEYS
        cached = getproperty(metadata, key)
        current = getproperty(params, key)
        agrees = cached isa Integer ?
            cached == current :
            isapprox(cached, current; atol=0, rtol=1e-12)
        agrees || push!(
            mismatches,
            "$key: cache=$cached script=$current",
        )
    end
    return mismatches
end

function load_or_build_tester_memory_process_tensor(params)
    force_rebuild = get(ENV, "PT_TESTER_REBUILD", "0") == "1"
    if force_rebuild
        println("PT_TESTER_REBUILD=1; constructing a new ACE process tensor.")
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
            mismatches = cache_parameter_mismatch(payload.metadata, params)
            if isempty(mismatches)
                println(
                    "Found a matching ACE cache; skipping process-tensor construction.",
                )
                println("  cache file:                  $(params.cache_path)")
                @printf(
                    "  maximum PT bond dimension:   %d\n",
                    payload.metadata.maxlinkdim,
                )
                return payload
            end
            println(
                "Cached process tensor does not match the script parameters; rebuilding.",
            )
            for line in mismatches
                println("  $line")
            end
        end
    else
        println("No process-tensor cache at $(params.cache_path); building.")
    end

    build_seconds = @elapsed begin
        payload = build_tester_memory_process_tensor(params)
    end
    save_tester_memory_cache(params.cache_path, payload)
    @printf("  ACE build time:              %.3f s\n", build_seconds)
    @printf("  wrote cache:                 %s\n", params.cache_path)
    @printf("  maximum PT bond dimension:   %d\n", payload.metadata.maxlinkdim)
    return payload
end

# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

function print_section(title::AbstractString)
    println()
    println(title)
    println("-"^length(title))
end

function density_matrix(state)
    pairs = siteinds(state)
    physical_sites = Index[
        only(filter(index -> plev(index) == 0, pair))
        for pair in pairs
    ]
    tensor = foldl(*, state)
    dimension = prod(dim.(physical_sites))
    return reshape(
        ComplexF64.(Array(
            tensor,
            prime.(physical_sites)...,
            physical_sites...,
        )),
        dimension,
        dimension,
    )
end

function expectation(state, operator)
    rho = density_matrix(state)
    return real(tr(operator * rho) / tr(rho))
end

function mutual_information(rho_system, rho_tester, rho_joint)
    function entropy(rho)
        normalized = (rho + rho') / (2real(tr(rho)))
        probabilities = clamp.(real.(eigvals(Hermitian(normalized))), 0, Inf)
        probabilities ./= sum(probabilities)
        return -sum(p * log2(p) for p in probabilities if p > eps(Float64))
    end
    return entropy(rho_system) + entropy(rho_tester) - entropy(rho_joint)
end

function store_retrieve_schedule(
    process_tensor,
    swap_gate,
    phase_gate,
    system_sites,
    tester_sites,
    store_step,
    phase_step,
    retrieve_step,
)
    1 <= store_step < phase_step < retrieve_step <= process_tensor.nsteps || throw(
        ArgumentError(
            "Expected store_step < phase_step < retrieve_step within the process.",
        ),
    )
    controls = TesterSeq(nsteps=process_tensor.nsteps)
    add!(
        controls,
        joint_unitary(swap_gate, system_sites, tester_sites),
        store_step,
    )
    add!(
        controls,
        tester_unitary(phase_gate, tester_sites),
        phase_step,
    )
    add!(
        controls,
        joint_unitary(swap_gate, system_sites, tester_sites),
        retrieve_step,
    )
    return controls
end

step_at_time(time, dt, nsteps) =
    clamp(round(Int, time / dt) + 1, 1, nsteps)

const SWAP_MARKER_COLOR = (0.72, 0.12, 0.18, 0.95)

function mark_swap_times!(axis, times)
    vlines!(
        axis,
        times;
        color=SWAP_MARKER_COLOR,
        linestyle=:dot,
        linewidth=4.5,
    )
    return nothing
end

function mark_phase_time!(axis, time)
    vlines!(
        axis,
        [time];
        color=(:gray35, 0.9),
        linestyle=:dash,
        linewidth=4.0,
    )
    return nothing
end

function annotate_protocol_times!(axis, store_t, phase_t, retrieve_t, y_label)
    text!(
        axis,
        store_t,
        y_label;
        text="SWAP",
        color=SWAP_MARKER_COLOR,
        align=(:left, :top),
        fontsize=16,
        offset=Vec2f(8, 0),
    )
    text!(
        axis,
        retrieve_t,
        y_label;
        text="SWAP",
        color=SWAP_MARKER_COLOR,
        align=(:left, :top),
        fontsize=16,
        offset=Vec2f(8, 0),
    )
    text!(
        axis,
        phase_t,
        y_label;
        text="Z gate",
        color=(:gray20, 0.95),
        align=(:left, :top),
        fontsize=16,
        offset=Vec2f(8, 0),
    )
    return nothing
end

# -----------------------------------------------------------------------------
# Cached process tensor and protocol times
# -----------------------------------------------------------------------------

const store_time_target = 1.0
const readout_delay_target = 0.50
const trace_warning_tolerance = 1e-3
const trace_assert_tolerance = 5e-2

output_dir = joinpath(@__DIR__, "figures")
mkpath(output_dir)
figure_png = joinpath(output_dir, "noisy_quantum_circuit_tester.png")

# -----------------------------------------------------------------------------
# Physical model and reusable process tensor
# -----------------------------------------------------------------------------

print_section("Testers and noisy quantum qubits")
println("One ACE process tensor is reused for many memory-control circuits.")

params = tester_memory_parameters()
print_tester_memory_parameters(params)
payload = load_or_build_tester_memory_process_tensor(params)
process_tensor = payload.process_tensor
system_sites = payload.system_sites
dt = payload.metadata.dt
final_time = payload.metadata.final_time
nsteps = payload.metadata.nsteps

println(process_tensor)

rho_system_0 = to_dm(MPS(system_sites, ["+"]))
tester_sites = siteinds("Qubit", 1)
rho_tester_0 = to_dm(MPS(tester_sites, ["0"]))
memory = tester(tester_sites, rho_tester_0)

# -----------------------------------------------------------------------------
# Baseline, spectator, and one representative memory circuit
# -----------------------------------------------------------------------------

print_section("Reference and store–wait–retrieve trajectories")

baseline = evolve(process_tensor, rho_system_0; progress=true)
spectator = evolve(
    process_tensor,
    rho_system_0;
    tester=memory,
    progress=true,
)

X_system = ComplexF64.(Array(
    op("X", only(system_sites)),
    prime(only(system_sites)),
    only(system_sites),
))
Y_system = ComplexF64.(Array(
    op("Y", only(system_sites)),
    prime(only(system_sites)),
    only(system_sites),
))
X_tester = ComplexF64.(Array(
    op("X", only(tester_sites)),
    prime(only(tester_sites)),
    only(tester_sites),
))

x_baseline = [expectation(rho, X_system) for rho in baseline.states_hilbert]
x_spectator = [expectation(rho, X_system) for rho in spectator.states_hilbert]
spectator_error = maximum(abs.(x_spectator .- x_baseline))
@printf("  identity-tester max |Δ<X_S>|: %.3e\n", spectator_error)

store_step = step_at_time(store_time_target, dt, nsteps)
retrieve_time_target = min(
    store_time_target + 2.0,
    final_time - readout_delay_target,
)
retrieve_step = step_at_time(retrieve_time_target, dt, nsteps)
phase_step = store_step + (retrieve_step - store_step) ÷ 2

swap_gate = op("SWAP", only(system_sites), only(tester_sites))
phase_gate = op("Z", only(tester_sites))
controls = store_retrieve_schedule(
    process_tensor,
    swap_gate,
    phase_gate,
    system_sites,
    tester_sites,
    store_step,
    phase_step,
    retrieve_step,
)

controlled = evolve(
    process_tensor,
    rho_system_0;
    tester=memory,
    tester_seq=controls,
    return_joint=true,
    progress=true,
)

x_system = [expectation(rho, X_system) for rho in controlled.states_hilbert]
x_tester = [
    expectation(rho, X_tester)
    for rho in controlled.tester_states_hilbert
]
information = [
    mutual_information(
        density_matrix(controlled.states_hilbert[k]),
        density_matrix(controlled.tester_states_hilbert[k]),
        density_matrix(controlled.joint_states_hilbert[k]),
    )
    for k in eachindex(controlled.times)
]

@printf("  store time:                  %.3f\n", controlled.times[store_step])
@printf("  phase-tag time:              %.3f\n", controlled.times[phase_step])
@printf("  retrieve time:               %.3f\n", controlled.times[retrieve_step])
@printf("  <X_A> after store:           %+.6f\n", x_tester[store_step])
@printf("  <X_A> after phase tag:       %+.6f\n", x_tester[phase_step])
@printf("  <X_S> after retrieve:        %+.6f\n", x_system[retrieve_step])
@printf("  maximum I(S:A):              %.6e bits\n", maximum(information))

# -----------------------------------------------------------------------------
# Idle-time sweep using the same process tensor
# -----------------------------------------------------------------------------

print_section("Idle-time sweep")

readout_steps = max(1, round(Int, readout_delay_target / dt))
minimum_idle_steps = max(2, round(Int, 0.35 / dt))
idle_stride = max(1, round(Int, 0.20 / dt))
maximum_idle_steps = nsteps - store_step - readout_steps - 1

maximum_idle_steps >= minimum_idle_steps || error(
    "Simulation window is too short for the requested idle-time sweep.",
)

idle_steps = collect(minimum_idle_steps:idle_stride:maximum_idle_steps)
idle_times = Float64[]
retrieval_coherence = Float64[]
baseline_coherence = Float64[]

report_stride = max(1, cld(length(idle_steps), 8))
for (iteration, idle_duration) in enumerate(idle_steps)
    retrieve = store_step + idle_duration
    phase = store_step + idle_duration ÷ 2
    readout = retrieve + readout_steps

    schedule = store_retrieve_schedule(
        process_tensor,
        swap_gate,
        phase_gate,
        system_sites,
        tester_sites,
        store_step,
        phase,
        retrieve,
    )
    trajectory = evolve(
        process_tensor,
        rho_system_0;
        tester=memory,
        tester_seq=schedule,
        return_tester=false,
        return_joint=false,
        progress=false,
    )

    controlled_x = expectation(trajectory.states_hilbert[readout], X_system)
    controlled_y = expectation(trajectory.states_hilbert[readout], Y_system)
    baseline_x = expectation(baseline.states_hilbert[readout], X_system)
    baseline_y = expectation(baseline.states_hilbert[readout], Y_system)

    push!(idle_times, idle_duration * dt)
    push!(retrieval_coherence, hypot(controlled_x, controlled_y))
    push!(baseline_coherence, hypot(baseline_x, baseline_y))

    if iteration == 1 || iteration == length(idle_steps) ||
       iteration % report_stride == 0
        @printf(
            "  %2d/%2d | idle %.3f | retrieved %.6f | bare %.6f\n",
            iteration,
            length(idle_steps),
            last(idle_times),
            last(retrieval_coherence),
            last(baseline_coherence),
        )
    end
end

# -----------------------------------------------------------------------------
# Numerical checks
# -----------------------------------------------------------------------------

print_section("Numerical checks")

trace_errors = Float64[]
for states in (
    baseline.states_hilbert,
    controlled.states_hilbert,
    controlled.tester_states_hilbert,
    controlled.joint_states_hilbert,
)
    append!(trace_errors, [abs(tr(density_matrix(rho)) - 1) for rho in states])
end

rho_joint_final = density_matrix(controlled.joint_states_hilbert[end])
joint_array = reshape(rho_joint_final, 2, 2, 2, 2)
rho_system_from_joint = zeros(ComplexF64, 2, 2)
rho_tester_from_joint = zeros(ComplexF64, 2, 2)
for tester_index in 1:2
    rho_system_from_joint .+= @view joint_array[:, tester_index, :, tester_index]
end
for system_index in 1:2
    rho_tester_from_joint .+= @view joint_array[system_index, :, system_index, :]
end

system_reduction_error = norm(
    rho_system_from_joint - density_matrix(controlled.states_hilbert[end]),
)
tester_reduction_error = norm(
    rho_tester_from_joint - density_matrix(controlled.tester_states_hilbert[end]),
)

@printf("  maximum trace error:         %.3e\n", maximum(trace_errors))
@printf("  final system reduction:      %.3e\n", system_reduction_error)
@printf("  final tester reduction:      %.3e\n", tester_reduction_error)

if maximum(trace_errors) > trace_warning_tolerance
    @warn(
        "ACE compression can violate exact trace preservation; " *
        "this is a truncation remainder, not a tester-contraction bug.",
        max_trace_error=maximum(trace_errors),
        trace_warning_tolerance=trace_warning_tolerance,
    )
end

@assert spectator_error < 1e-8
@assert length(controlled.times) == length(controlled.states_hilbert)
@assert length(controlled.times) == length(controlled.tester_states_hilbert)
@assert length(controlled.times) == length(controlled.joint_states_hilbert)
@assert maximum(trace_errors) < trace_assert_tolerance
@assert system_reduction_error < 1e-7
@assert tester_reduction_error < 1e-7
@assert all(isfinite, x_baseline)
@assert all(isfinite, x_system)
@assert all(isfinite, x_tester)
@assert all(isfinite, information)
@assert minimum(information) > -1e-8
@assert all(value -> abs(value) <= 1 + 1e-3, x_system)
@assert all(isfinite, retrieval_coherence)

# -----------------------------------------------------------------------------
# Three-panel figure
# -----------------------------------------------------------------------------

print_section("Generating figure")

figure = Figure(size=(1120, 1200), fontsize=20)

coherence_axis = Axis(
    figure[1, 1],
    xlabel=L"t",
    ylabel=L"\langle X\rangle",
    title="Store, phase-tag, and retrieve the qubit state",
)
lines!(
    coherence_axis,
    baseline.times,
    x_baseline;
    linewidth=2.5,
    linestyle=:dash,
    label=L"S\ \mathrm{without\ tester}",
)
lines!(
    coherence_axis,
    controlled.times,
    x_system;
    linewidth=3,
    label=L"S\ \mathrm{controlled}",
)
lines!(
    coherence_axis,
    controlled.times,
    x_tester;
    linewidth=3,
    label=L"A\ \mathrm{memory}",
)
mark_phase_time!(coherence_axis, controlled.times[phase_step])
mark_swap_times!(
    coherence_axis,
    controlled.times[[store_step, retrieve_step]],
)
axislegend(coherence_axis; position=:lb, framevisible=false)
coherence_hi = maximum((
    maximum(x_baseline),
    maximum(x_system),
    maximum(x_tester),
))
coherence_lo = minimum((
    minimum(x_baseline),
    minimum(x_system),
    minimum(x_tester),
))
coherence_span = coherence_hi - coherence_lo
coherence_ymin = coherence_lo - 0.06 * coherence_span
coherence_ymax = coherence_hi + 0.16 * coherence_span
ylims!(coherence_axis, coherence_ymin, coherence_ymax)
annotate_protocol_times!(
    coherence_axis,
    controlled.times[store_step],
    controlled.times[phase_step],
    controlled.times[retrieve_step],
    coherence_ymin + 0.96 * (coherence_ymax - coherence_ymin),
)

information_axis = Axis(
    figure[2, 1],
    xlabel=L"t",
    ylabel=L"I(S{:}A)\ \mathrm{[bits]}",
    title="System–memory correlations",
)
lines!(
    information_axis,
    controlled.times,
    information;
    linewidth=3,
)
mark_phase_time!(information_axis, controlled.times[phase_step])
mark_swap_times!(
    information_axis,
    controlled.times[[store_step, retrieve_step]],
)
information_hi = maximum(information)
information_lo = minimum(information)
information_span = max(information_hi - information_lo, 1e-3)
information_ymin = information_lo - 0.08 * information_span
information_ymax = information_hi + 0.20 * information_span
ylims!(information_axis, information_ymin, information_ymax)
annotate_protocol_times!(
    information_axis,
    controlled.times[store_step],
    controlled.times[phase_step],
    controlled.times[retrieve_step],
    information_ymin + 0.96 * (information_ymax - information_ymin),
)

idle_axis = Axis(
    figure[3, 1],
    xlabel=L"\tau_{\mathrm{idle}}",
    ylabel=L"C_{xy}",
    title="Coherence read a fixed delay after retrieval",
)
scatterlines!(
    idle_axis,
    idle_times,
    retrieval_coherence;
    linewidth=3,
    marker=:circle,
    label=L"\mathrm{store/retrieve}",
)
scatterlines!(
    idle_axis,
    idle_times,
    baseline_coherence;
    linewidth=2.5,
    linestyle=:dash,
    marker=:rect,
    label=L"\mathrm{bare\ noisy\ qubit}",
)
axislegend(idle_axis; position=:lb, framevisible=false)

save(figure_png, figure)

println()
println("Saved:")
println("  $figure_png")
println("Done.")
println(
    "The idle-time curve is a memory-sensitive control observable, " *
    "not a universal non-Markovianity measure.",
)
