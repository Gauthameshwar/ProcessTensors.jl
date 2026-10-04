# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: scripts/thermal_spinboson_ace.jl
# Contributor: Gauthameshwar S.
#
# Driven thermal spin-boson model of Cygorek and Gauger,
# J. Chem. Phys. 161, 074111 (2024), Fig. 3(c,f), plus a hotter-bath comparison.
#
# Run with:
# julia --project=. -t auto scripts/thermal_spinboson_ace.jl
# Place this file in the repository's scripts/ directory.
# PT_THERMAL_CACHE overrides the cache path; PT_ACE_REBUILD=1 forces rebuilding.
# Force rebuilding after changing model code or package versions.

# --- Parameters: physical frequencies in ps^-1, times in ps, ħ = 1 ---
const N_bath = 60
const local_dim = 5
const Ω = 3.0
const ω_min, ω_max = 0.0, 30.0
const spectral_strength, spectral_cutoff = 0.2, 3.0
const thermal_frequency = 1.5       # θ = k_B T / ħ; not temperature in kelvin
const hot_thermal_frequency = 6.0
const dt, final_time = 0.05, 5.0
const nsteps = round(Int, final_time / dt) + 1
const ace_cutoff, ace_maxdim = 1e-5, 512
const ace_compression = :canonzip
const trace_warning_tolerance = 1e-5
const population_tolerance = 2e-3
@assert N_bath > 0 && local_dim >= 2 && dt > 0 && final_time > 0
@assert 0 <= ω_min < ω_max && spectral_strength >= 0 && spectral_cutoff > 0
@assert all(θ -> isfinite(θ) && θ >= 0, (thermal_frequency, hot_thermal_frequency))
@assert isapprox((nsteps - 1) * dt, final_time; atol=100eps(Float64))

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
figure_path = joinpath(output_dir, "thermal_spinboson_ace.png")
cache_path = get(ENV, "PT_THERMAL_CACHE", joinpath(@__DIR__, ".cache", "thermal_spinboson_ace.jls"))
mkpath(output_dir)
mkpath(dirname(cache_path))

# --- Midpoint quadrature: g_k² = J(ω_k) Δω ---
spectral_density(ω) = spectral_strength * ω * exp(-ω / spectral_cutoff)
Δω = (ω_max - ω_min) / N_bath
frequencies = [ω_min + (k - 0.5) * Δω for k in 1:N_bath]
couplings = sqrt.(spectral_density.(frequencies) .* Δω)
@assert all(>(0), frequencies) && all(isfinite, couplings)
temperatures = (thermal_frequency, hot_thermal_frequency)

# These tails belong to the infinite free-oscillator Gibbs distribution.
# They diagnose initial Fock truncation, not the eventual system-observable error.
for θ in temperatures
    tail = θ == 0 ? 0.0 : maximum(exp.(-local_dim .* frequencies ./ θ))
    println((thermal_frequency=θ, lowest_mode=first(frequencies),
             largest_omitted_Gibbs_weight=tail))
    tail > 0.01 && @warn "Check local_dim convergence: appreciable initial Gibbs weight is omitted" θ tail
end
println((bath_modes=N_bath, local_dim=local_dim, spacing=Δω,
         formal_bath_Hilbert_dimension=BigInt(local_dim)^N_bath,
         formal_bath_Liouville_dimension=BigInt(local_dim)^(2N_bath)))

# --- Three utilities: mode assembly, cache reading, population diagnostics ---
function thermal_bath(frequencies, couplings, θ, local_dim)
    sites = siteinds("Boson", length(frequencies); dim=local_dim)
    liouville_sites = liouv_sites(sites)
    modes = BosonicMode[]
    for k in eachindex(frequencies)
        ωk, gk = frequencies[k], couplings[k]
        H_mode = OpSum()
        H_mode += ωk, "N", 1
        coupling = OpSum()
        coupling += gk, "A", 1, "ProjUp", 2
        coupling += gk, "Adag", 1, "ProjUp", 2
        coupling += gk^2 / ωk, "ProjUp", 2 # Counterterm g_k^2 A^2/ω_k; A^2=A.
        push!(modes, thermal_mode([liouville_sites[k]], H_mode, θ; coupling=coupling))
    end
    # Suppress only the bath-size warning for joint Dense construction.
    return with_logger(() -> bosonic_bath(modes), NullLogger())
end

function read_cache(path, params)
    get(ENV, "PT_ACE_REBUILD", "0") == "1" && return nothing
    isfile(path) || return nothing
    try
        payload = deserialize(path)
        payload.metadata.format == 2 || return nothing
        matches = all(k -> hasproperty(payload.metadata, k) &&
                      getproperty(payload.metadata, k) == getproperty(params, k), keys(params))
        matches || return nothing
        all(k -> hasproperty(payload, k), (:process_tensor_T1, :process_tensor_T3, :system_sites)) || return nothing
        return payload
    catch err
        err isa InterruptException && rethrow()
        @warn "Unreadable cache; rebuilding" path exception=err
        return nothing
    end
end

function population_diagnostics(trajectory)
    population, trace_errors, hermiticity_errors = Float64[], Float64[], Float64[]
    for state in trajectory.states_hilbert
        tensor = foldl(*, state)
        site = only(filter(i -> plev(i) == 0 && hastags(i, "Site"), inds(tensor)))
        ρ = ComplexF64.(Array(tensor, prime(site), site))
        z = tr(ρ)
        isfinite(z) && abs(z) > 1e-12 || error("Nonfinite or vanishing trace")
        push!(population, real(ρ[1, 1] / z)) # Up=e; retain raw trace diagnostics.
        push!(trace_errors, abs(z - 1))
        push!(hermiticity_errors, norm(ρ - ρ') / norm(ρ))
    end
    @assert all(p -> isfinite(p) && -population_tolerance <= p <= 1 + population_tolerance, population)
    maximum(trace_errors) > trace_warning_tolerance && @warn "Trace drift exceeds tolerance" max_trace_error=maximum(trace_errors)
    return (; times=trajectory.times, population, trace_errors, hermiticity_errors)
end

# --- Isolated spin: package trajectory and analytical Rabi reference ---
system_sites = siteinds("S=1/2", 1)
H_system = OpSum()
H_system += Ω, "Sx", 1
system = spin_system(system_sites, H_system)
initial_density = to_dm(MPS(system_sites, ["Dn"]))
closed_build_time = @elapsed closed_process = build_process_tensor(
    system; dt=dt, nsteps=nsteps, sys_alg=Trotter{2}(), progress=false,
)
closed_evolution_time = @elapsed closed_trajectory = evolve(closed_process, initial_density)
closed = population_diagnostics(closed_trajectory)
closed_analytic_error = maximum(abs.(closed.population .- sin.(Ω .* closed.times ./ 2).^2))
println((isolated_analytic_error=closed_analytic_error,
         build_seconds=closed_build_time, evolution_seconds=closed_evolution_time))

# --- Build or reload both thermal process tensors ---
params = (; N_bath, local_dim, Ω, ω_min, ω_max, thermal_frequency, hot_thermal_frequency,
          dt, final_time, nsteps, ace_cutoff, ace_maxdim,
          spectral_strength, spectral_cutoff, ace_compression)
payload = read_cache(cache_path, params)
cache_hit = payload !== nothing
build_times = zeros(length(temperatures))
if !cache_hit
    processes = ProcessTensor[]
    for (i, θ) in enumerate(temperatures)
        println((building_thermal_frequency=θ,))
        started = time_ns()
        bath = thermal_bath(frequencies, couplings, θ, local_dim)
        pt = build_process_tensor(
            system; environment=bath, dt=dt, nsteps=nsteps,
            method=ACE(cutoff=ace_cutoff, maxdim=ace_maxdim, compression=ace_compression),
            sys_alg=Trotter{2}(), combine_alg=Trotter{2}(), progress=true, verbose=false,
        )
        build_times[i] = (time_ns() - started) / 1e9
        # Keep the temporal cores and system; omit the original bath from disk.
        push!(processes, ProcessTensor(pt.core, pt.system, nothing, pt.dt, pt.nsteps, pt.coupling_site))
        println((thermal_frequency=θ, build_seconds=build_times[i], max_pt_bond=maxlinkdim(pt)))
    end
    # Legacy field names are retained for compatibility; T3 means the hotter case.
    payload = (; process_tensor_T1=processes[1], process_tensor_T3=processes[2], system_sites,
               metadata=(; format=2, params...,
                         maxlinkdim=maximum(maxlinkdim.(processes)),
                         maxlinkdim_T1=maxlinkdim(processes[1]), maxlinkdim_T3=maxlinkdim(processes[2])))
    serialize(cache_path * ".tmp", payload)
    mv(cache_path * ".tmp", cache_path; force=true)
end

# --- Evaluate both baths on their cached system indices ---
processes = (payload.process_tensor_T1, payload.process_tensor_T3)
initial_density = to_dm(MPS(payload.system_sites, ["Dn"]))
results = NamedTuple[]
for (θ, pt) in zip(temperatures, processes)
    evolution_time = @elapsed trajectory = evolve(pt, initial_density)
    data = population_diagnostics(trajectory)
    max_bond = maxlinkdim(pt)
    max_bond >= ace_maxdim && @warn "ACE reached the bond cap; check convergence" θ max_bond
    push!(results, (; data..., thermal_frequency=θ, max_pt_bond=max_bond, evolution_time))
    println((thermal_frequency=θ, cache_hit=cache_hit, final_population=last(data.population),
             max_trace_error=maximum(data.trace_errors),
             max_hermiticity_error=maximum(data.hermiticity_errors),
             max_pt_bond=max_bond, evolution_seconds=evolution_time))
end

# --- Figure: the shared spectral density and the three population curves ---
figure = Figure(size=(1100, 850), fontsize=18)
spectral_axis = Axis(figure[1, 1]; xlabel="ω (ps⁻¹)", ylabel="J(ω) (ps⁻¹)",
                     title="Ohmic environment and its $N_bath-mode discretisation")
ω_plot = range(ω_min, ω_max; length=800)
lines!(spectral_axis, ω_plot, spectral_density.(ω_plot); color=:steelblue,
       linewidth=2.6, label="J(ω) = $(spectral_strength) ω exp(−ω/$(spectral_cutoff))")
scatter!(spectral_axis, frequencies, spectral_density.(frequencies); marker=:circle,
         markersize=12, color=:lightskyblue, strokecolor=:black, strokewidth=2.4, label="ACE modes")
axislegend(spectral_axis; position=:rt)

population_axis = Axis(figure[2, 1]; xlabel="t (ps)", ylabel="Pe(t)",
                      title="Thermal damping of Rabi oscillations (M=$local_dim levels)")
lines!(population_axis, closed.times, closed.population; color=:gray35,
       linewidth=2.5, linestyle=:dash, label="isolated TLS")
for (r, color) in zip(results, (:steelblue, :darkorange))
    lines!(population_axis, r.times, r.population; color=color, linewidth=2.6,
           label="kᵦT/ħ = $(r.thermal_frequency) ps⁻¹ (ACE)")
end
ylims!(population_axis, -0.03, 1.03)
axislegend(population_axis; position=:rt)
rowgap!(figure.layout, 18)
save(figure_path, figure)
println("Saved figure: $figure_path")
println((ace_build_seconds=sum(build_times),
         max_trace_error=max(maximum(closed.trace_errors),
                             maximum(maximum(r.trace_errors) for r in results)),
         closed_analytic_error=closed_analytic_error))

# Loss of contrast and Pe≈1/2 alone do not establish a Gibbs or maximally mixed state.
# Converge local_dim (especially at high T), mode grid/window, dt, and ACE truncation.
