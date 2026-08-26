# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: scripts/central_spin_ace.jl
# Contributor: Gauthameshwar S.
#
# Reproduces the fully polarized central-spin model of Cygorek et al.,
# Nature Physics 18, 662–668 (2022), Fig. 4a, with ProcessTensors.jl.
#
#   H_S = 0
#   H_E = sum_k J_k (Sx sx_k + Sy sy_k + Sz sz_k),  J_k = J / N
#   J = ħ = 1
#   dt = 0.01
#   t_final = 20
#   ACE threshold ε = 1e-10
#   ACE compression = zip-up (forward truncation on the 2001-step chain)
#   bath initial state: every spin along +z
#
# The paper reports d_max = 4 for N = 10, 100, and 1000, and the reduced
# dynamics approach (1/2) cos(t/2) as N → ∞.
#
# Run with:
#   julia --project=. scripts/central_spin_ace.jl
#
# Matching ACE caches in scripts/.cache skip process-tensor construction.
# Override the cache directory with PT_CENTRAL_CACHE_DIR, or force a rebuild
# with PT_ACE_REBUILD=1.

import Pkg

const REPO_ROOT = dirname(@__DIR__)
const _PLOT_ENV = joinpath(@__DIR__, ".plot_examples_env")

function activate_plot_examples_env!()
    mkpath(_PLOT_ENV)
    Pkg.activate(_PLOT_ENV)
    manifest = joinpath(_PLOT_ENV, "Manifest.toml")

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

activate_plot_examples_env!()

using Logging
using LinearAlgebra
using Printf
using Serialization
using CairoMakie
using ITensors
using ITensors.Ops: Trotter
using LaTeXStrings
using ProcessTensors

CairoMakie.activate!()

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

function save_ace_cache(path, payload)
    mkpath(dirname(path))
    open(path, "w") do io
        serialize(io, payload)
    end
    return path
end

function cache_parameter_mismatch(metadata, params, keys, format)
    mismatches = String[]
    hasproperty(metadata, :format) || return ["format: missing"]
    metadata.format == format || return [
        "format: cache=$(metadata.format) script=$format",
    ]
    for key in keys
        cached = getproperty(metadata, key)
        current = getproperty(params, key)
        agrees = if cached isa Integer && current isa Integer
            cached == current
        elseif cached isa Number && current isa Number
            isapprox(cached, current; atol=0, rtol=1e-12)
        else
            cached == current
        end
        agrees || push!(mismatches, "$key: cache=$cached script=$current")
    end
    return mismatches
end

function load_or_build_ace_cache(
    path,
    params,
    keys,
    format,
    builder;
    label::AbstractString,
)
    force_rebuild = get(ENV, "PT_ACE_REBUILD", "0") == "1"
    if force_rebuild
        println("PT_ACE_REBUILD=1; constructing $label.")
    elseif isfile(path)
        payload = try
            open(deserialize, path)
        catch err
            @warn "Could not read the process-tensor cache; rebuilding." exception = (
                err,
                catch_backtrace(),
            )
            nothing
        end
        if payload !== nothing
            mismatches = cache_parameter_mismatch(payload.metadata, params, keys, format)
            if isempty(mismatches)
                println("Found a matching ACE cache; skipping process-tensor construction.")
                println("  cache file:                  $path")
                if hasproperty(payload.metadata, :maxlinkdim)
                    @printf(
                        "  maximum PT bond dimension:   %d\n",
                        payload.metadata.maxlinkdim,
                    )
                end
                return payload, 0.0
            end
            println("Cached process tensor does not match the script parameters; rebuilding.")
            for line in mismatches
                println("  $line")
            end
        end
    else
        println("No process-tensor cache at $path; building.")
    end

    build_seconds = @elapsed begin
        payload = builder()
    end
    save_ace_cache(path, payload)
    @printf("  ACE build time:              %.3f s\n", build_seconds)
    @printf("  wrote cache:                 %s\n", path)
    if hasproperty(payload.metadata, :maxlinkdim)
        @printf("  maximum PT bond dimension:   %d\n", payload.metadata.maxlinkdim)
    end
    return payload, build_seconds
end

# ------------------------------------------------------------------------------
# 1. Small script utilities
# ------------------------------------------------------------------------------

const STATUS_WIDTH = 100

function print_section(title::AbstractString)
    println()
    println(title)
    println("-"^length(title))
end

function update_status(message::AbstractString)
    print("\r", rpad(message, STATUS_WIDTH))
    flush(stdout)
end

function finish_status(message::AbstractString="")
    print("\r", " "^STATUS_WIDTH, "\r")
    isempty(message) || println(message)
    flush(stdout)
end

function one_site_density_matrix(ρ)
    T = foldl(*, ρ)
    site = only(filter(i -> plev(i) == 0 && hastags(i, "Site"), inds(T)))
    return ComplexF64.(Array(T, prime(site), site))
end

function uniform_sample_indices(n::Int; nmarkers::Int)
    nmarkers = clamp(nmarkers, 1, n)
    return unique(round.(Int, range(1, n; length=nmarkers)))
end

function central_spin_diagnostics(trajectory, Sx_matrix)
    sx = Float64[]
    trace_errors = Float64[]
    hermiticity_errors = Float64[]

    for ρ in trajectory.states_hilbert
        ρ_matrix = one_site_density_matrix(ρ)
        trace_value = tr(ρ_matrix)
        ρ_norm = norm(ρ_matrix)
        push!(sx, real(tr(Sx_matrix * ρ_matrix) / trace_value))
        push!(trace_errors, abs(trace_value - 1))
        push!(
            hermiticity_errors,
            ρ_norm == 0 ? 0.0 : norm(ρ_matrix - ρ_matrix') / ρ_norm,
        )
    end

    analytical_sx = 0.5 .* cos.(trajectory.times ./ 2)
    ed_errors = abs.(sx .- analytical_sx)
    return sx, trace_errors, hermiticity_errors, ed_errors, analytical_sx
end

function polarized_spin_density(physical_site, liouville_site)
    return to_liouville(
        to_dm(MPS([physical_site], ["Up"]));
        sites=[liouville_site],
    )
end

function polarized_central_spin_bath(N_bath::Int; J::Real)
    Jk = J / N_bath
    bath_sites = siteinds("S=1/2", N_bath)
    bath_liouville_sites = liouv_sites(bath_sites)
    modes = SpinMode[]

    with_logger(NullLogger()) do
        for k in 1:N_bath
            update_status("  preparing polarized bath mode $k / $N_bath")

            coupling = OpSum()
            coupling += Jk, "Sx", 1, "Sx", 2
            coupling += Jk, "Sy", 1, "Sy", 2
            coupling += Jk, "Sz", 1, "Sz", 2

            push!(
                modes,
                spin_mode(
                    [bath_liouville_sites[k]],
                    OpSum(),
                    polarized_spin_density(bath_sites[k], bath_liouville_sites[k]);
                    coupling=coupling,
                ),
            )
        end
    end

    finish_status("  prepared $N_bath polarized bath modes")
    bath = with_logger(NullLogger()) do
        spin_bath(modes)
    end
    return bath
end

# ------------------------------------------------------------------------------
# 2. User-adjustable parameters
# ------------------------------------------------------------------------------

const J = 1.0
const dt = 0.01
const final_time = 20.0
const nsteps = round(Int, final_time / dt) + 1
const ace_cutoff = 1e-10
const ace_maxdim = 1024
const ace_compression = :zipup
const N_bath_values = [5, 10, 100, 1000]
const published_polarized_rank = 4
const trace_warning_tolerance = 1e-4
const spin_bound_tolerance = 1e-4
const ed_nmarkers = 21
const log_error_floor = 1e-16
const line_colors = [:dodgerblue, :darkorange, :seagreen, :mediumpurple]

output_dir = joinpath(@__DIR__, "figures")
mkpath(output_dir)
figure_path = joinpath(output_dir, "central_spin_ace.png")
cache_dir = get(
    ENV,
    "PT_CENTRAL_CACHE_DIR",
    joinpath(@__DIR__, ".cache"),
)
const CENTRAL_CACHE_FORMAT = 1
const CENTRAL_CACHE_KEYS = (
    :N_bath,
    :J,
    :dt,
    :final_time,
    :nsteps,
    :ace_cutoff,
    :ace_maxdim,
    :ace_compression,
)

# ------------------------------------------------------------------------------
# 3. Physical problem
# ------------------------------------------------------------------------------

print_section("Cygorek polarized central-spin benchmark")

println("Reproducing Fig. 4a of Cygorek et al.: fully polarized bath, N sweep.")
println("  H_S:                          0")
println("  bath free Hamiltonians:       0")
println("  bath initial state:           all spins along +z")
println("  interaction:                  J_k (Sx sx + Sy sy + Sz sz)")
println("  coupling per mode:            J_k = J / N")
@printf("  J = ħ:                        %.1f\n", J)
@printf("  dt:                           %.3f\n", dt)
@printf("  final time:                   %.1f\n", final_time)
@printf("  snapshots:                    %d\n", nsteps)
@printf("  ACE cutoff ε:                 %.1e\n", ace_cutoff)
@printf("  ACE maxdim safety cap:        %d\n", ace_maxdim)
println("  ACE compression:              $ace_compression")
println("  ACE mode maps:                Hilbert U = exp(-i H Δt), fused onto Liouville PT legs")
println("  N values:                     $(join(N_bath_values, ", "))")
println("  published polarized d_max:    $published_polarized_rank")

system_sites = siteinds("S=1/2", 1)
system = with_logger(NullLogger()) do
    spin_system(system_sites, OpSum())
end
initial_density = to_dm(MPS(system_sites, ["+"]))
Sx_matrix = ComplexF64.(
    Array(
        op("Sx", system_sites[1]),
        prime(system_sites[1]),
        system_sites[1],
    ),
)

# ------------------------------------------------------------------------------
# 4. ACE N sweep
# ------------------------------------------------------------------------------

print_section("ACE scaling sweep")

results = Dict{Int,NamedTuple}()

for N_bath in N_bath_values
    println()
    println("N = $N_bath")

    cache_path = joinpath(cache_dir, "central_spin_ace_N$(N_bath).jls")
    cache_params = (;
        N_bath,
        J,
        dt,
        final_time,
        nsteps,
        ace_cutoff,
        ace_maxdim,
        ace_compression,
    )
    payload, build_time = load_or_build_ace_cache(
        cache_path,
        cache_params,
        CENTRAL_CACHE_KEYS,
        CENTRAL_CACHE_FORMAT,
        () -> begin
            bath = polarized_central_spin_bath(N_bath; J=J)
            update_status("  building ACE PT: polarized, N=$N_bath")
            process_tensor = build_process_tensor(
                system;
                method=ACE(
                    cutoff=ace_cutoff,
                    maxdim=ace_maxdim,
                    compression=ace_compression,
                ),
                environment=bath,
                dt=dt,
                nsteps=nsteps,
                sys_alg=Trotter{2}(),
                combine_alg=Trotter{2}(),
            )
            slim = slim_process_tensor(process_tensor)
            return (;
                process_tensor=slim,
                system_sites,
                metadata=(;
                    format=CENTRAL_CACHE_FORMAT,
                    N_bath,
                    J,
                    dt,
                    final_time,
                    nsteps,
                    ace_cutoff,
                    ace_maxdim,
                    ace_compression,
                    maxlinkdim=maxlinkdim(slim),
                ),
            )
        end;
        label="polarized central-spin ACE process tensor (N=$N_bath)",
    )

    process_tensor = payload.process_tensor
    open_system_sites = payload.system_sites
    open_initial_density = to_dm(MPS(open_system_sites, ["+"]))
    open_Sx_matrix = ComplexF64.(
        Array(
            op("Sx", open_system_sites[1]),
            prime(open_system_sites[1]),
            open_system_sites[1],
        ),
    )

    finish_status(
        @sprintf("  ACE PT ready: polarized N=%4d", N_bath),
    )

    bond_dimensions = Int[d for d in linkdims(process_tensor) if d !== nothing]
    isempty(bond_dimensions) && error("Process tensor has no temporal link dimensions.")
    max_bond_dimension = maximum(bond_dimensions)

    @printf(
        "    J_k = %.6f, max(linkdims(PT)) = %d\n",
        J / N_bath,
        max_bond_dimension,
    )

    if max_bond_dimension >= ace_maxdim
        @warn "ACE reached the maxdim safety cap." N_bath=N_bath max_bond_dimension=max_bond_dimension
    end
    if max_bond_dimension != published_polarized_rank
        @warn "Published fully polarized benchmark reports d_max=$published_polarized_rank." N_bath=N_bath max_bond_dimension=max_bond_dimension
    end

    update_status("  evolving polarized PT, N=$N_bath")
    evolution_time = @elapsed begin
        trajectory = evolve(process_tensor, open_initial_density)
    end
    finish_status(
        @sprintf("  trajectory evolved: polarized N=%4d in %.3f s", N_bath, evolution_time),
    )

    sx, trace_errors, hermiticity_errors, ed_errors, _ =
        central_spin_diagnostics(trajectory, open_Sx_matrix)
    max_trace_error = maximum(trace_errors)
    max_hermiticity_error = maximum(hermiticity_errors)
    max_spin_bound_excess = max(maximum(abs, sx) - 0.5, 0.0)
    analytical_error = maximum(ed_errors)

    @printf(
        "    max |tr ρ-1| = %.3e, max ‖ρ-ρ†‖/‖ρ‖ = %.3e, max |Sx|-1/2 = %.3e, max |Sx - ED| = %.3e\n",
        max_trace_error,
        max_hermiticity_error,
        max_spin_bound_excess,
        analytical_error,
    )

    if max_trace_error > trace_warning_tolerance
        @warn "Trace drift exceeds the benchmark warning tolerance." N_bath=N_bath max_trace_error=max_trace_error
    end
    max_spin_bound_excess <= spin_bound_tolerance || error(
        "Unphysical central-spin trajectory for N=$N_bath: max |Sx|-1/2 = $max_spin_bound_excess.",
    )

    results[N_bath] = (
        max_bond_dimension=max_bond_dimension,
        bond_dimensions=bond_dimensions,
        build_time=build_time,
        evolution_time=evolution_time,
        times=trajectory.times,
        sx=sx,
        trace_errors=trace_errors,
        hermiticity_errors=hermiticity_errors,
        ed_errors=ed_errors,
        max_trace_error=max_trace_error,
        max_hermiticity_error=max_hermiticity_error,
        max_spin_bound_excess=max_spin_bound_excess,
        analytical_error=analytical_error,
    )
end

# ------------------------------------------------------------------------------
# 5. Figure
# ------------------------------------------------------------------------------

print_section("Plotting")

# Extra width is for the right-hand legends so the stacked panels keep
# the previous 1100 × 850 plot aspect.
figure = Figure(size=(1300, 850), fontsize=18)
times = results[last(N_bath_values)].times
ed_sx = 0.5 .* cos.(times ./ 2)
ed_idx = uniform_sample_indices(length(times); nmarkers=ed_nmarkers)

dynamics_axis = Axis(
    figure[1, 1];
    ylabel=L"$\langle S_x\rangle/\hbar$",
    title="Fully polarized central-spin dynamics",
    xticklabelsvisible=false,
    xticksvisible=false,
    xlabelvisible=false,
)

for (i, N_bath) in enumerate(N_bath_values)
    trajectory = results[N_bath]
    lines!(
        dynamics_axis,
        trajectory.times,
        trajectory.sx;
        color=line_colors[i],
        linewidth=2.2,
        label="N = $N_bath",
    )
end

scatter!(
    dynamics_axis,
    times[ed_idx],
    ed_sx[ed_idx];
    marker=:x,
    markersize=14,
    color=:black,
    label=L"$N \to \infty$",
)

ylims!(dynamics_axis, -0.55, 0.55)

Legend(
    figure[1, 2],
    dynamics_axis;
    labelsize=16,
    nbanks=1,
    tellheight=false,
    valign=:center,
    halign=:left,
)

error_axis = Axis(
    figure[2, 1];
    xlabel=L"$tJ/\hbar$",
    ylabel="density-matrix error",
    yscale=log10,
    title="Trace, Hermiticity, and ED errors",
)

for (i, N_bath) in enumerate(N_bath_values)
    trajectory = results[N_bath]
    color = line_colors[i]
    lines!(
        error_axis,
        trajectory.times,
        max.(trajectory.ed_errors, log_error_floor);
        color=color,
        linestyle=:solid,
        linewidth=1.8,
    )
    lines!(
        error_axis,
        trajectory.times,
        max.(trajectory.trace_errors, log_error_floor);
        color=color,
        linestyle=:dash,
        linewidth=1.8,
    )
    lines!(
        error_axis,
        trajectory.times,
        max.(trajectory.hermiticity_errors, log_error_floor);
        color=color,
        linestyle=:dot,
        linewidth=1.8,
    )
end

Legend(
    figure[2, 2],
    [
        LineElement(color=:gray40, linestyle=:solid, linewidth=2),
        LineElement(color=:gray40, linestyle=:dash, linewidth=2),
        LineElement(color=:gray40, linestyle=:dot, linewidth=2),
    ],
    [
        L"$|\langle S_x\rangle-\frac{1}{2}\cos(t/2)|$",
        L"$|\mathrm{tr}\,\rho-1|$",
        L"$\Vert\rho-\rho^\dagger\Vert/\Vert\rho\Vert$",
    ];
    labelsize=16,
    nbanks=1,
    tellheight=false,
    valign=:center,
    halign=:left,
)

linkxaxes!(dynamics_axis, error_axis)
rowgap!(figure.layout, 12)
colgap!(figure.layout, 12)
rowsize!(figure.layout, 1, Relative(0.5))
rowsize!(figure.layout, 2, Relative(0.5))

save(figure_path, figure)

println("Saved figure:")
println("  $figure_path")

# ------------------------------------------------------------------------------
# 6. Final summary
# ------------------------------------------------------------------------------

print_section("Summary")

println("Completed polarized Cygorek central-spin ACE benchmark.")
println("  figure:                        $figure_path")

ranks = [results[N].max_bond_dimension for N in N_bath_values]
analytical_errors = [results[N].analytical_error for N in N_bath_values]

for N_bath in N_bath_values
    result = results[N_bath]
    @printf(
        "  N=%4d  J_k=%8.5f  chi_max=%4d  |trρ-1|_max=%.3e  ‖ρ-ρ†‖_max=%.3e  |Sx-ED|_max=%.3e  build=%8.3f s  evolve=%8.3f s\n",
        N_bath,
        J / N_bath,
        result.max_bond_dimension,
        result.max_trace_error,
        result.max_hermiticity_error,
        result.analytical_error,
        result.build_time,
        result.evolution_time,
    )
end

println()
println("Sanity checks:")
println("  published d_max = $published_polarized_rank for polarized N = 10, 100, 1000")
println("  observed chi_max = $(join(ranks, ", "))")
println("  analytical errors should decrease with N: $(join(round.(analytical_errors; sigdigits=3), ", "))")
