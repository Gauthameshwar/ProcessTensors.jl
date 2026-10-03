# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: scripts/noisy_quantum_circuit_tester.jl
# Contributor: Gauthameshwar S.
#
# Reuses one thermal-bath process tensor for a baseline, an idle tester, and
# the SWAP–Z–SWAP protocol, then writes the two-panel figure.
#
# Run with:
# julia --project=. scripts/noisy_quantum_circuit_tester.jl [--rebuild]

# --- User parameters: bath, compression, and the three protocol events ---
const N_bath = 24
const local_dim = 3
const alpha = 0.008
const omega_cutoff = 4.0
const omega_max = 20.0
const thermal_frequency = 2.5
const dt = 0.20
const final_time = 5.0
const nsteps = round(Int, final_time / dt) + 1
const ace_cutoff = 1e-5
const ace_maxdim = 512
const store_time_target, phase_time_target, retrieve_time_target = 1.0, 2.0, 3.0
const trace_warning_tolerance, trace_assert_tolerance = 1e-3, 5e-2
@assert N_bath > 0 && local_dim >= 2 && alpha >= 0 && omega_max > 0 && dt > 0
@assert thermal_frequency >= 0 && isfinite(thermal_frequency)
@assert isapprox((nsteps - 1) * dt, final_time; atol=100eps(Float64))
@assert 0 < store_time_target < phase_time_target < retrieve_time_target <= final_time

# --- Automatic plot environment, as in the original script ---
import Pkg
plot_env = joinpath(@__DIR__, ".plot_examples_env")
Pkg.activate(plot_env)
if !isfile(joinpath(plot_env, "Manifest.toml"))
    Pkg.develop(Pkg.PackageSpec(path=dirname(@__DIR__)))
    Pkg.add(["CairoMakie", "LaTeXStrings"])
else
    Pkg.resolve()
    Pkg.instantiate()
end

using LinearAlgebra
using Logging
using Serialization
using CairoMakie
using ITensors
using ITensors.Ops: Trotter
using ProcessTensors
CairoMakie.activate!()

if !(isempty(ARGS) || ARGS == ["--rebuild"])
    error("Usage: julia --project=. scripts/noisy_quantum_circuit_tester.jl [--rebuild]")
end
force_rebuild = "--rebuild" in ARGS
cache_path = joinpath(@__DIR__, ".cache", "noisy_quantum_circuit_tester_pt.jls")
figure_path = joinpath(@__DIR__, "figures", "noisy_quantum_circuit_tester.png")
mkpath(dirname(cache_path))
mkpath(dirname(figure_path))
params = (; N_bath, local_dim, alpha, omega_cutoff, omega_max, thermal_frequency,
          dt, final_time, nsteps, ace_cutoff, ace_maxdim)

# --- Cache reader: fail closed on missing/mismatched metadata ---
function read_cache(path, params)
    isfile(path) || return nothing
    try
        payload = deserialize(path)
        if payload.metadata.format != 1
            @warn "Cache format is not reusable; rebuilding" path
            return nothing
        end
        matches = all(k -> hasproperty(payload.metadata, k) &&
                      getproperty(payload.metadata, k) == getproperty(params, k), keys(params))
        if !matches
            @warn "Cached process tensor does not match these parameters; rebuilding" path
            return nothing
        end
        hasproperty(payload, :process_tensor) && hasproperty(payload, :system_sites) || return nothing
        return payload
    catch err
        err isa InterruptException && rethrow()
        @warn "Unreadable cache; rebuilding" path exception=err
        return nothing
    end
end

# --- Construct the noise once; controls are not part of the cache key ---
payload = force_rebuild ? nothing : read_cache(cache_path, params)
cache_hit = payload !== nothing
build_seconds = 0.0
if !cache_hit
    started = time_ns()
    system_sites = siteinds("Qubit", 1)
    system = qubit_system(system_sites)
    Δω = omega_max / N_bath
    frequencies = [(k - 0.5) * Δω for k in 1:N_bath]
    couplings = sqrt.(2alpha .* frequencies .* exp.(-frequencies ./ omega_cutoff) .* Δω)
    bath_sites = siteinds("Boson", N_bath; dim=local_dim)
    bath_liouville_sites = liouv_sites(bath_sites)
    modes = BosonicMode[]
    for k in 1:N_bath
        H_mode = OpSum() + (frequencies[k], "N", 1)
        coupling = OpSum()
        coupling += couplings[k], "A", 1, "Z", 2
        coupling += couplings[k], "Adag", 1, "Z", 2
        push!(modes, thermal_mode([bath_liouville_sites[k]], H_mode,
                                 thermal_frequency; coupling=coupling))
    end
    bath = with_logger(() -> bosonic_bath(modes), NullLogger())
    pt = build_process_tensor(
        system; environment=bath, dt=dt, nsteps=nsteps,
        method=ACE(cutoff=ace_cutoff, maxdim=ace_maxdim),
        sys_alg=Trotter{2}(), combine_alg=Trotter{2}(), progress=true,
    )
    # Store cores and system, omitting the original bath from the payload.
    slim = ProcessTensor(pt.core, pt.system, nothing, pt.dt, pt.nsteps, pt.coupling_site)
    payload = (; process_tensor=slim, system_sites,
               metadata=(; format=1, params..., maxlinkdim=maxlinkdim(slim)))
    serialize(cache_path * ".tmp", payload)
    mv(cache_path * ".tmp", cache_path; force=true)
    build_seconds = (time_ns() - started) / 1e9
end
process_tensor, system_sites = payload.process_tensor, payload.system_sites
println((cache_hit=cache_hit, build_seconds=build_seconds, max_pt_bond=maxlinkdim(process_tensor)))
maxlinkdim(process_tensor) >= ace_maxdim && @warn "ACE reached the bond cap; check convergence"
# Free-oscillator initial Gibbs tail omitted by the local Fock cutoff.
lowest_frequency = omega_max / (2N_bath)
thermal_tail = thermal_frequency == 0 ? 0.0 : exp(-local_dim * lowest_frequency / thermal_frequency)
println((largest_omitted_Gibbs_weight=thermal_tail,))

# --- Explicit tester circuit: SWAP to store, Z to phase-tag, SWAP to retrieve ---
rho_Q0 = to_dm(MPS(system_sites, ["+"]))
ancilla_sites = siteinds("Qubit", 1)
memory = tester(ancilla_sites, to_dm(MPS(ancilla_sites, ["0"])))
store_step, phase_step, retrieve_step = round.(Int, [store_time_target, phase_time_target, retrieve_time_target] ./ dt) .+ 1
@assert 1 <= store_step < phase_step < retrieve_step <= nsteps
swap_gate = op("SWAP", only(system_sites), only(ancilla_sites))
phase_gate = op("Z", only(ancilla_sites))
controls = TesterSeq(nsteps=nsteps)
add!(controls, joint_unitary(swap_gate, system_sites, ancilla_sites), store_step)
add!(controls, tester_unitary(phase_gate, ancilla_sites), phase_step)
add!(controls, joint_unitary(swap_gate, system_sites, ancilla_sites), retrieve_step)

# Joint controls apply half-gates on each side of a noisy slab; they are not
# instantaneous SWAPs. Report the actual grid labels after rounding target times.
baseline_seconds = @elapsed baseline = evolve(process_tensor, rho_Q0; progress=true)
spectator_seconds = @elapsed spectator = evolve(process_tensor, rho_Q0; tester=memory, progress=true)
controlled_seconds = @elapsed controlled = evolve(
    process_tensor, rho_Q0; tester=memory, tester_seq=controls, return_joint=true, progress=true,
)
event_steps = [store_step, phase_step, retrieve_step]
event_times = controlled.times[event_steps]
println((store_time=event_times[1], phase_time=event_times[2], retrieve_time=event_times[3],
         baseline_seconds=baseline_seconds, spectator_seconds=spectator_seconds,
         controlled_seconds=controlled_seconds))

# --- Two utilities for all one- and two-qubit outputs ---
function density_matrix(state)
    sites = [only(filter(i -> plev(i) == 0, pair)) for pair in siteinds(state)]
    tensor = foldl(*, state)
    d = prod(dim.(sites))
    return reshape(ComplexF64.(Array(tensor, prime.(sites)..., sites...)), d, d)
end

function entropy_diagnostics(ρ; tolerance=1e-7)
    z = tr(ρ)
    isfinite(z) && real(z) > 1e-12 || error("Invalid density-matrix trace")
    hermiticity_error = norm(ρ - ρ') / norm(ρ)
    λ = eigvals(Hermitian((ρ + ρ') / (2real(z))))
    min_eigenvalue = minimum(λ)
    valid = min_eigenvalue >= -tolerance && hermiticity_error <= tolerance
    entropy = NaN
    if valid
        p = max.(λ, 0) # Only negativity within the declared tolerance is clipped.
        p ./= sum(p)
        entropy = -sum(x * log2(x) for x in p if x > 0)
    end
    return (; entropy, min_eigenvalue, hermiticity_error, trace_error=abs(z - 1), valid)
end

rho_Q = density_matrix.(controlled.states_hilbert)
rho_A = density_matrix.(controlled.tester_states_hilbert)
rho_QA = density_matrix.(controlled.joint_states_hilbert)
rho_baseline = density_matrix.(baseline.states_hilbert)
rho_spectator = density_matrix.(spectator.states_hilbert)
X = ComplexF64[0 1; 1 0]
x_Q = [real(tr(X * ρ) / tr(ρ)) for ρ in rho_Q]
x_A = [real(tr(X * ρ) / tr(ρ)) for ρ in rho_A]
x_baseline = [real(tr(X * ρ) / tr(ρ)) for ρ in rho_baseline]
Q_data, A_data, QA_data = entropy_diagnostics.(rho_Q), entropy_diagnostics.(rho_A), entropy_diagnostics.(rho_QA)
information = [q.entropy + a.entropy - qa.entropy for (q, a, qa) in zip(Q_data, A_data, QA_data)]
checks = vcat(Q_data, A_data, QA_data, entropy_diagnostics.(rho_baseline), entropy_diagnostics.(rho_spectator))
max_trace_error = maximum(c.trace_error for c in checks)
min_eigenvalue = minimum(c.min_eigenvalue for c in checks)
spectator_error = maximum(norm.(rho_spectator .- rho_baseline))
invalid_entropy_inputs = count(c -> !c.valid, checks)

# The returned joint basis has Q as its first (fastest) local index.
# Compare its final partial traces with the separately returned marginals.
joint = reshape(last(rho_QA), 2, 2, 2, 2)
Q_from_joint = joint[:, 1, :, 1] + joint[:, 2, :, 2]
A_from_joint = joint[1, :, 1, :] + joint[2, :, 2, :]
reduction_errors = (processor=norm(Q_from_joint - last(rho_Q)), ancilla=norm(A_from_joint - last(rho_A)))
println((identity_tester_error=spectator_error, reduction_errors=reduction_errors,
         max_trace_error=max_trace_error, minimum_eigenvalue=min_eigenvalue,
         max_hermiticity_error=maximum(c.hermiticity_error for c in checks),
         invalid_entropy_inputs=invalid_entropy_inputs))
max_trace_error > trace_warning_tolerance && @warn "Trace drift exceeds tolerance; check compression and numerical settings" max_trace_error
invalid_entropy_inputs > 0 && @warn "Invalid entropy inputs; affected mutual-information samples are NaN and appear as gaps" invalid_entropy_inputs
@assert spectator_error < 1e-8
@assert max_trace_error < trace_assert_tolerance
@assert maximum(values(reduction_errors)) < 1e-7
@assert all(isfinite, x_Q) && all(isfinite, x_A) && all(isfinite, x_baseline)
@assert maximum(abs, x_Q) <= 1 + 1e-3 && maximum(abs, x_A) <= 1 + 1e-3

for (event, k) in zip((:store, :phase, :retrieve), event_steps)
    println((event=event, time=controlled.times[k], x_Q=x_Q[k], x_A=x_A[k],
             ancilla_coherence=2abs(rho_A[k][1, 2] / tr(rho_A[k])), mutual_information=information[k]))
end

# --- Two-panel figure; no fixed-delay readout or idle-time sweep ---
figure = Figure(size=(1120, 850), fontsize=20)
signal_axis = Axis(figure[1, 1]; ylabel="⟨X⟩", title="Store, phase-tag, and retrieve",
                   xticklabelsvisible=false)
lines!(signal_axis, baseline.times, x_baseline; linewidth=2.5, linestyle=:dash, label="Q without tester")
lines!(signal_axis, controlled.times, x_Q; linewidth=3, label="Q controlled")
lines!(signal_axis, controlled.times, x_A; linewidth=3, label="A memory")
axislegend(signal_axis; position=:lb, framevisible=false)
ylims!(signal_axis, -1.05, 1.22)

information_axis = Axis(figure[2, 1]; xlabel="t", ylabel="I(Q:A) [bits]",
                        title="Processor–ancilla correlations")
lines!(information_axis, controlled.times, information; linewidth=3)
for axis in (signal_axis, information_axis)
    vlines!(axis, event_times[[1, 3]]; color=:crimson, linestyle=:dot, linewidth=2.5)
    vlines!(axis, [event_times[2]]; color=:gray40, linestyle=:dash, linewidth=2.5)
end
for (t, label) in zip(event_times, ("SWAP", "Z", "SWAP"))
    text!(signal_axis, t, 1.12; text=label, fontsize=16, align=(:left, :center), offset=Vec2f(7, 0))
end
linkxaxes!(signal_axis, information_axis)
rowgap!(figure.layout, 16)
save(figure_path, figure)
println("Saved figure: $figure_path")
# Nonzero mutual information is not an entanglement or non-Markovianity witness.
# The ideal instantaneous-SWAP limit differs from these finite-slab joint gates.
