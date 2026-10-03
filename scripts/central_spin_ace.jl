# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: scripts/central_spin_ace.jl
# Contributor: Gauthameshwar S.
#
# Fully polarised central-spin benchmark: H = (J/N) Σ_k S⋅s_k, ħ=1.
# Central spin initially +x, bath spins +z, no free Hamiltonians.
# Based on Cygorek et al., Nature Physics 18, 662–668 (2022), Fig. 4a.
#
# Run with:
# julia --project=. -t auto scripts/central_spin_ace.jl
#
# Set PT_ACE_REBUILD=1 after changing model code or package versions.
# PT_CENTRAL_CACHE_DIR overrides the cache directory.

# --- Parameters: start here when exploring ---
const J = 1.0
const dt = 0.01
const final_time = 20.0
const nsteps = round(Int, final_time / dt) + 1
const N_bath_values = [5, 10, 100, 1000] # Use [5, 10] for a shorter first run.
const ace_cutoff = 1e-10
const ace_maxdim = 1024
const ace_compression = :zipup_cpp
const trace_warning_tolerance = 1e-4
const spin_bound_tolerance = 1e-4
const reference_nmarkers = 21
const log_error_floor = 1e-16
const line_colors = [:dodgerblue, :darkorange, :seagreen, :mediumpurple]
@assert dt > 0 && J > 0 && all(N -> N > 0, N_bath_values)
@assert !isempty(N_bath_values) && isapprox((nsteps - 1) * dt, final_time)

# --- Plot environment (same automatic setup as the original script) ---
import Pkg
const REPO_ROOT = dirname(@__DIR__)
plot_env = joinpath(@__DIR__, ".plot_examples_env")
Pkg.activate(plot_env)
if !isfile(joinpath(plot_env, "Manifest.toml"))
    Pkg.develop(Pkg.PackageSpec(path=REPO_ROOT))
    Pkg.add(["CairoMakie", "LaTeXStrings"])
else
    Pkg.resolve()
    Pkg.instantiate()
end

using Logging
using LinearAlgebra
using Serialization
using CairoMakie
using ITensors
using ITensors.Ops: Trotter
using ProcessTensors
CairoMakie.activate!()

output_dir = joinpath(@__DIR__, "figures")
cache_dir = get(ENV, "PT_CENTRAL_CACHE_DIR", joinpath(@__DIR__, ".cache"))
mkpath(output_dir)
mkpath(cache_dir)
figure_path = joinpath(output_dir, "central_spin_ace.png")

# --- Three small utilities: bath assembly, cache reading, trajectory diagnostics ---
# Only constructors with intentional zero Hamiltonians have their warnings silenced.
function polarised_bath(N, J)
    sites = siteinds("S=1/2", N)
    liouville_sites = liouv_sites(sites)
    coupling = OpSum()
    coupling += J / N, "Sx", 1, "Sx", 2
    coupling += J / N, "Sy", 1, "Sy", 2
    coupling += J / N, "Sz", 1, "Sz", 2
    modes = SpinMode[]
    for k in 1:N
        ρ = to_dm(MPS([sites[k]], ["Up"]))
        ρ_l = to_liouville(ρ; sites=[liouville_sites[k]])
        mode = with_logger(() -> spin_mode([liouville_sites[k]], OpSum(), ρ_l;
                                           coupling=copy(coupling)), NullLogger())
        push!(modes, mode)
    end
    return spin_bath(modes)
end

# Early returns keep cache handling separate from the visible construction call.
# Compare all construction settings; corrupted or incompatible caches rebuild.
function read_cache(path, params)
    get(ENV, "PT_ACE_REBUILD", "0") == "1" && return nothing
    isfile(path) || return nothing
    try
        payload = deserialize(path)
        metadata = payload.metadata
        metadata.format == 2 || return nothing
        matches = all(k -> hasproperty(metadata, k) &&
                      getproperty(metadata, k) == getproperty(params, k), keys(params))
        matches || return nothing
        hasproperty(payload, :process_tensor) && hasproperty(payload, :system_sites) || return nothing
        return payload
    catch err
        err isa InterruptException && rethrow()
        @warn "Unreadable cache; rebuilding" path exception=err
        return nothing
    end
end

function trajectory_diagnostics(trajectory, N, J)
    sx, trace_errors, hermiticity_errors = Float64[], Float64[], Float64[]
    Sx = ComplexF64[0 1; 1 0] / 2
    for state in trajectory.states_hilbert
        tensor = foldl(*, state)
        site = only(filter(i -> plev(i) == 0 && hastags(i, "Site"), inds(tensor)))
        ρ = ComplexF64.(Array(tensor, prime(site), site))
        z = tr(ρ)
        isfinite(z) && abs(z) > 1e-12 || error("Nonfinite or vanishing trace")
        push!(sx, real(tr(Sx * ρ) / z))
        push!(trace_errors, abs(z - 1))
        push!(hermiticity_errors, norm(ρ - ρ') / norm(ρ))
    end
    times = trajectory.times
    exact_sx = (1 .+ N .* cos.(J * (N + 1) .* times ./ (2N))) ./ (2(N + 1))
    limiting_sx = 0.5 .* cos.(J .* times ./ 2)
    return (; times, sx, trace_errors, hermiticity_errors, exact_sx,
            finite_N_errors=abs.(sx .- exact_sx),
            large_N_deviations=abs.(sx .- limiting_sx))
end

# --- Construct (or reload) each process, then evaluate it ---
results = Dict{Int,NamedTuple}()
for N_bath in N_bath_values
    println((bath_spins=N_bath, coupling=J / N_bath, dt=dt, final_time=final_time))
    cache_path = joinpath(cache_dir, "central_spin_ace_N$(N_bath).jls")
    params = (; N_bath, J, dt, final_time, nsteps, ace_cutoff, ace_maxdim, ace_compression)
    payload = read_cache(cache_path, params)
    build_time = 0.0
    cache_hit = payload !== nothing

    if !cache_hit
        system_sites = siteinds("S=1/2", 1)
        system = with_logger(() -> spin_system(system_sites, OpSum()), NullLogger())
        build_time = @elapsed begin
            bath = polarised_bath(N_bath, J)
            pt = build_process_tensor(
                system; environment=bath, dt=dt, nsteps=nsteps,
                method=ACE(cutoff=ace_cutoff, maxdim=ace_maxdim, compression=ace_compression),
                sys_alg=Trotter{2}(), combine_alg=Trotter{2}(),
            )
        end
        # Keep the system and temporal cores; omit the original bath from the cache.
        slim = ProcessTensor(pt.core, pt.system, nothing, pt.dt, pt.nsteps, pt.coupling_site)
        payload = (; process_tensor=slim, system_sites,
                   metadata=(; format=2, params..., maxlinkdim=maxlinkdim(slim)))
        serialize(cache_path * ".tmp", payload)
        mv(cache_path * ".tmp", cache_path; force=true)
    end

    # Cached Index identities must be reused for the initial preparation.
    process_tensor = payload.process_tensor
    initial_density = to_dm(MPS(payload.system_sites, ["+"]))
    evolution_time = @elapsed trajectory = evolve(process_tensor, initial_density)
    diagnostics = trajectory_diagnostics(trajectory, N_bath, J)
    bonds = Int[d for d in linkdims(process_tensor) if d !== nothing]
    max_bond = maxlinkdim(process_tensor)
    max_trace_error = maximum(diagnostics.trace_errors)
    max_spin_bound_excess = max(maximum(abs, diagnostics.sx) - 0.5, 0.0)

    @assert all(isfinite, diagnostics.sx)
    @assert max_spin_bound_excess <= spin_bound_tolerance "Transverse spin exceeds its physical bound"
    max_trace_error > trace_warning_tolerance && @warn "Trace drift exceeds tolerance" N_bath max_trace_error
    max_bond >= ace_maxdim && @warn "ACE reached the bond cap; check convergence" N_bath max_bond
    results[N_bath] = (; diagnostics..., bond_dimensions=bonds, max_bond_dimension=max_bond,
                       build_time, evolution_time, cache_hit, max_trace_error, max_spin_bound_excess)
    println((cache_hit=cache_hit, max_bond=max_bond, build_seconds=build_time,
             evolution_seconds=evolution_time,
             max_trace_error=max_trace_error,
             max_hermiticity_error=maximum(diagnostics.hermiticity_errors),
             max_finite_N_error=maximum(diagnostics.finite_N_errors),
             max_large_N_deviation=maximum(diagnostics.large_N_deviations)))
end

# --- Plot: physical finite-size effects and numerical diagnostics ---
# All reference curves use the actual returned times and the adjustable J.
# Clipping to log_error_floor is for display only; stored diagnostics are raw.
figure = Figure(size=(1400, 900), fontsize=18)
dynamics_axis = Axis(figure[1, 1]; ylabel="⟨Sx⟩ / ħ",
                     title="Fully polarised central-spin dynamics",
                     xticklabelsvisible=false, xticksvisible=false)
error_axis = Axis(figure[2, 1]; xlabel="tJ / ħ", ylabel="absolute / relative deviation",
                  yscale=log10, title="Error against the exact finite-N result")

for (i, N) in enumerate(N_bath_values)
    r = results[N]
    color = line_colors[mod1(i, length(line_colors))]
    time_axis = J .* r.times
    lines!(dynamics_axis, time_axis, r.sx; color=color, linewidth=2.2, label="N = $N")
    lines!(error_axis, time_axis, max.(r.finite_N_errors, log_error_floor);
           color=color, linestyle=:solid, linewidth=1.8)
    lines!(error_axis, time_axis, max.(r.trace_errors, log_error_floor);
           color=color, linestyle=:dash, linewidth=1.8)
    lines!(error_axis, time_axis, max.(r.hermiticity_errors, log_error_floor);
           color=color, linestyle=:dot, linewidth=1.8)
end

times = results[last(N_bath_values)].times
marker_indices = unique(round.(Int, range(1, length(times);
                              length=clamp(reference_nmarkers, 1, length(times)))))
scatter!(dynamics_axis, J .* times[marker_indices], 0.5 .* cos.(J .* times[marker_indices] ./ 2);
         marker=:x, markersize=14, color=:black, label="N → ∞")
ylims!(dynamics_axis, -0.55, 0.55)
Legend(figure[1, 2], dynamics_axis; labelsize=16, tellheight=false)
Legend(figure[2, 2],
       [LineElement(color=:gray40, linestyle=s, linewidth=2) for s in (:solid, :dash, :dot)],
       ["|⟨Sx⟩ − exact finite-N|", "|tr ρ − 1|", "‖ρ − ρ†‖ / ‖ρ‖"];
       labelsize=16, tellheight=false)
linkxaxes!(dynamics_axis, error_axis)
rowgap!(figure.layout, 12)
colgap!(figure.layout, 12)
save(figure_path, figure)
println("Saved figure: $figure_path")

# The published d_max=4 (N=10,100,1000) is a comparison point, not an invariant
# of every representation/compression setting. Report bonds without asserting it.
println((bath_sizes=N_bath_values,
         maximum_bonds=[results[N].max_bond_dimension for N in N_bath_values]))
# Small trace/Hermiticity defects or a physical Sx do not prove positivity.
# For convergence, vary dt/cutoff/maxdim at fixed N and inspect finite_N_errors.
