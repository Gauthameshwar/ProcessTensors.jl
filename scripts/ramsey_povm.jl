# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: scripts/ramsey_povm.jl
# Contributor: Gauthameshwar S.
#
# Three unsharp Ramsey readouts, each followed by the same active reset.
#
# Run with:
# julia --project=. -t auto scripts/ramsey_povm.jl
# PT_RAMSEY_CACHE overrides the cache path; PT_RAMSEY_REBUILD=1 forces rebuilding.
# Force rebuilding after changing bath construction or package versions.

# --- Edit the experiment here (ħ = k_B = 1) ---
const N_BATH, LOCAL_DIM = 4, 3
const ALPHA, OMEGA_C, OMEGA_MAX = 0.20, 4.0, 20.0
const TEMPERATURE = 2.5
const DT, NSTEPS = 0.15, 14
const ACE_CUTOFF, ACE_MAXDIM = 1e-5, 256
const ACE_COMPRESSION = :zipup_cpp
const ETA, OUTCOMES = 0.90, (-1, 1)
const READOUT_STEPS = (4, 8, 12)
@assert N_BATH > 0 && LOCAL_DIM >= 2 && ALPHA >= 0
@assert OMEGA_C > 0 && OMEGA_MAX > 0 && TEMPERATURE >= 0 && DT > 0
@assert 0 <= ETA <= 1 && ACE_CUTOFF > 0 && ACE_MAXDIM > 0
@assert length(READOUT_STEPS) == 3 && issorted(READOUT_STEPS)
@assert length(unique(READOUT_STEPS)) == 3
@assert all(s -> 0 < s < NSTEPS, READOUT_STEPS)

# --- Automatic plotting environment ---
import Pkg
plot_env = joinpath(@__DIR__, ".plot_examples_env")
Pkg.activate(plot_env)
if !isfile(joinpath(plot_env, "Manifest.toml"))
    Pkg.develop(Pkg.PackageSpec(path=dirname(@__DIR__)))
    Pkg.add(["CairoMakie", "LaTeXStrings"])
else
    Pkg.instantiate()
end

using CairoMakie
using ITensors
using ITensors.Ops: Trotter
using Logging
using Serialization
using ProcessTensors
CairoMakie.activate!()

# --- Reuse a matching process tensor, including its original site indices ---
# Visibility and readout times do not affect the bath PT, so are not cache keys.
parameters = (; N_bath=N_BATH, local_dim=LOCAL_DIM, alpha=ALPHA,
    omega_cutoff=OMEGA_C, omega_max=OMEGA_MAX, thermal_frequency=TEMPERATURE,
    dt=DT, nsteps=NSTEPS, ace_cutoff=ACE_CUTOFF, ace_maxdim=ACE_MAXDIM,
    ace_compression=ACE_COMPRESSION)
cache_path = get(ENV, "PT_RAMSEY_CACHE", joinpath(@__DIR__, ".cache", "ramsey_povm_pt.jls"))

function read_cache(path, parameters)
    get(ENV, "PT_RAMSEY_REBUILD", "0") == "1" && return nothing
    isfile(path) || return nothing
    try
        payload = open(deserialize, path)
        get(payload.metadata, :format, 0) == 2 || return nothing
        all(k -> get(payload.metadata, k, nothing) == parameters[k], keys(parameters)) || return nothing
        return payload
    catch err
        @warn "Could not read the PT cache; rebuilding" exception=(err, catch_backtrace())
        return nothing
    end
end

payload = read_cache(cache_path, parameters)
cache_hit = payload !== nothing
if !cache_hit
    system_sites = siteinds("Qubit", 1)
    system = with_logger(() -> qubit_system(system_sites), NullLogger())
    # H_E = Σ ω_k b†_k b_k; H_SE = Z Σ g_k(b_k+b†_k), with Pauli Z.
    # J(ω) = 2αω exp(-ω/ω_c), g_k² = J(ω_k) Δω; each mode starts thermal.
    Δω = OMEGA_MAX / N_BATH
    frequencies = [(k - 0.5) * Δω for k in 1:N_BATH]
    couplings = sqrt.(2ALPHA .* frequencies .* exp.(-frequencies ./ OMEGA_C) .* Δω)
    bath_sites = siteinds("Boson", N_BATH; dim=LOCAL_DIM)
    bath_liouville_sites = liouv_sites(bath_sites)
    modes = BosonicMode[]
    for k in eachindex(frequencies)
        H_mode = OpSum() + (frequencies[k], "N", 1)
        coupling = OpSum()
        coupling += couplings[k], "A", 1, "Z", 2
        coupling += couplings[k], "Adag", 1, "Z", 2
        push!(modes, thermal_mode([bath_liouville_sites[k]], H_mode, TEMPERATURE;
                                 coupling=coupling))
    end
    bath = with_logger(() -> bosonic_bath(modes), NullLogger())
    pt = build_process_tensor(
        system; method=ACE(cutoff=ACE_CUTOFF, maxdim=ACE_MAXDIM, compression=ACE_COMPRESSION),
        environment=bath, dt=DT, nsteps=NSTEPS,
        sys_alg=Trotter{2}(), combine_alg=Trotter{2}(), progress=false)
    # The contracted core no longer needs the bath object in the serialized cache.
    process_tensor = ProcessTensor(pt.core, pt.system, nothing, pt.dt, pt.nsteps, pt.coupling_site)
    metadata = (; format=2, parameters..., maxlinkdim=maxlinkdim(process_tensor))
    payload = (; process_tensor, system_sites, metadata)
    mkpath(dirname(cache_path))
    temporary_path = cache_path * ".tmp"
    open(io -> serialize(io, payload), temporary_path, "w")
    mv(temporary_path, cache_path; force=true)
end
process_tensor, system_sites = payload.process_tensor, payload.system_sites
println((cache_hit=cache_hit, maximum_bond_dimension=maxlinkdim(process_tensor)))

# --- Unsharp X readout followed by reset ---
# E_x = (I + x η X)/2; A_x(ρ) = Tr(E_x ρ) ρ_reset.
# In Liouville space A_x = |ρ_reset⟩⟩⟨⟨E_x|, not the reversed outer product.
# This resets the qubit, while the conditional bath state can retain the record.
rho_reset = to_dm(MPS(system_sites, ["+"]))
effects = Dict(x => OpSum() + (0.5, "Id", 1) + (x * ETA / 2, "X", 1)
               for x in OUTCOMES)
ramsey_instruments = Dict(
    x => observable_measurement(effects[x]) * state_preparation(rho_reset)
    for x in OUTCOMES)

# --- Joint probabilities and independent-record reference ---
# Slot s closes out_(s-1) after s propagations, then prepares the next input.
# The original schedule gives times 0.6, 1.2, 1.8: three equal waits.
readout_times = collect(READOUT_STEPS) .* DT
final_time = NSTEPS * DT
println((readout_times=readout_times, final_time=final_time, visibility=ETA))

function branch_probability(record)
    sequence = default_schedule(process_tensor)
    add!(sequence, state_preparation(rho_reset), 0)
    for (step, outcome) in zip(READOUT_STEPS, record)
        add!(sequence, ramsey_instruments[outcome], step)
    end
    add!(sequence, trace_out(), process_tensor.nsteps)
    value = evaluate_process(process_tensor, sequence; progress=false)
    @assert isfinite(value) && abs(imag(value)) < 1e-7
    return real(value)
end

records = vec(collect(Iterators.product(OUTCOMES, OUTCOMES, OUTCOMES)))
probabilities = branch_probability.(records)
normalization_error = abs(sum(probabilities) - 1)
@assert minimum(probabilities) >= -1e-7
@assert normalization_error < 5e-3

# q(x₁,x₂,x₃) = ∏ⱼ pⱼ(xⱼ): no assumption that the three marginals agree.
# For exact probabilities D = ½Σ|p-q| is total variation distance.
# Keep numerical weights raw; normalization/positivity errors remain visible.
# Columns contain P(-) and P(+) for each round; retain the raw PT weights.
marginals = zeros(3, 2)
for (record, probability) in zip(records, probabilities)
    for round in 1:3
        marginals[round, record[round] == -1 ? 1 : 2] += probability
    end
end
independent_probabilities = [
    prod(marginals[j, record[j] == -1 ? 1 : 2] for j in 1:3)
    for record in records]
factorization_residual = sum(abs.(probabilities .- independent_probabilities)) / 2
println((normalization_error=normalization_error, minimum_probability=minimum(probabilities),
         factorization_residual=factorization_residual))

labels = [join(x == 1 ? "+" : "-" for x in record) for record in records]
for (label, p, q) in zip(labels, probabilities, independent_probabilities)
    println((record=label, joint=p, independent=q))
end
normalization_error > 1e-5 && @warn "Check ACE convergence before interpreting the residual" normalization_error

# --- Two figures: persistent bath, then record statistics ---
set_theme!(Theme(fontsize=18,
    Axis=(topspinevisible=false, rightspinevisible=false, xgridvisible=false),
    Legend=(labelsize=16,)))
pt_color, effect_color, reset_color = "#3A9A5B", "#FCB06D", "#F7E58B"
protocol_figure = Figure(size=(1100, 300))
ax = Axis(protocol_figure[1, 1]; title="Ramsey readouts: reset the qubit, retain the bath",
          limits=(-0.13final_time, 1.07final_time, -0.5, 0.85))
hidedecorations!(ax)
hidespines!(ax)
lines!(ax, [0, final_time], [0.55, 0.55]; color=pt_color, linewidth=4)
lines!(ax, [0, final_time], [0, 0]; color=:black, linewidth=2)
text!(ax, -0.03final_time, 0.55; text="Bath", align=(:right, :center), color=pt_color)
text!(ax, -0.03final_time, 0; text="Qubit", align=(:right, :center))
scatter!(ax, [0], [0]; color=reset_color, markersize=18, strokewidth=1)
text!(ax, 0, -0.12; text="prepare +", align=(:center, :top))
# Offsets separate symbols visually; measurement and reset share one time slot.
marker_offset = 0.014final_time
for (j, time) in enumerate(readout_times)
    lines!(ax, [time, time], [0.08, 0.50]; color=(pt_color, 0.4), linewidth=2)
    scatter!(ax, [time-marker_offset, time+marker_offset], [0, 0];
             color=[effect_color, reset_color], markersize=18, strokewidth=1)
    text!(ax, time, 0.67; text="x$j", align=(:center, :center))
    text!(ax, time, -0.12; text="measure → reset
t=$(round(time; digits=3))",
          align=(:center, :top), fontsize=16)
end
text!(ax, final_time, -0.12; text="trace", align=(:center, :top))
Label(protocol_figure[2, 1], "X readout visibility η = $ETA  •  Every reset prepares |+⟩  •  Bath is never reset";
      fontsize=16)

record_figure = Figure(size=(1050, 420))
probability_axis = Axis(record_figure[1, 1]; title="Outcome-record probabilities",
    xlabel="(x₁ x₂ x₃)", ylabel="Probability", xticks=(1:8, labels))
positions = collect(1:8)
barplot!(probability_axis, positions .- 0.15, probabilities; width=0.26,
         color=pt_color, label="process-tensor record")
barplot!(probability_axis, positions .+ 0.15, independent_probabilities; width=0.26,
         color=effect_color, label="product of marginals")
axislegend(probability_axis; position=:rt, framevisible=false)
# Do not conceal a negative numerical weight with a hard zero lower limit.
ylims!(probability_axis, min(0, 1.1minimum(probabilities)),
        1.2maximum(vcat(probabilities, independent_probabilities)))
Label(record_figure[2, 1], "Factorization residual = $(round(factorization_residual; sigdigits=4))";
      fontsize=16)

output_dir = joinpath(@__DIR__, "figures")
mkpath(output_dir)
for (stem, figure) in (("ramsey_povm_protocol", protocol_figure), ("ramsey_povm_records", record_figure))
    save(joinpath(output_dir, stem * ".pdf"), figure)
    save(joinpath(output_dir, stem * ".png"), figure; px_per_unit=2)
end
println((figures=output_dir,))

# Try changing: test detector blindness, bath decoupling, or the memory timescale.
# ETA=0 gives eight equiprobable records; ALPHA=0 gives independent biased coins.
# For equal waits m*DT, use slots (m, 2m, 3m), NSTEPS > 3m.
# Converge DT, ACE_CUTOFF/ACE_MAXDIM, LOCAL_DIM, and N_BATH before assigning
# physical significance to a small residual. Factorization does not rule out
# memory that this particular instrument cannot detect.
