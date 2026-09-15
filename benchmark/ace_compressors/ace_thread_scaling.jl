# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: benchmark/ace_compressors/ace_thread_scaling.jl
# Contributor: Gauthameshwar S.
#
# Time zipup_cpp / canonzip on two fixed process tensors while the
# launcher varies OpenBLAS/MKL threads. Julia stays at `-t 1`.
#
#   ACE_BLAS_THREADS=1 ACE_BENCH_SAMPLES=1 \
#     julia -t 1 --project=. benchmark/ace_compressors/ace_thread_scaling.jl
#
# Smoke (tiny nsteps, zipup_cpp only):
#
#   ACE_THREAD_SMOKE=1 ACE_BLAS_THREADS=1 julia -t 1 --project=. \
#     benchmark/ace_compressors/ace_thread_scaling.jl

include(joinpath(@__DIR__, "ace_compression_common.jl"))

const CSV_PATH = results_path("ace_thread_scaling.csv")
const CSV_HEADER = (
    "case",
    "model",
    "strategy",
    "blas_threads",
    "julia_threads",
    "t_min_s",
    "t_median_s",
    "chi_max",
    "allocated_bytes",
    "allocs",
    "nsamples",
)

const SB_C = 0.2
const SB_GAMMA = 0.1
const SB_OMEGA_C = 1.0
const SB_OMEGA_MIN = 0.0
const SB_OMEGA_MAX = 7.5
const SB_OMEGA = 1.0

lorentzian_J(ω; C=SB_C, γ=SB_GAMMA, ωc=SB_OMEGA_C) =
    (C / π) * γ / ((ω - ωc)^2 + γ^2)

function lorentzian_mode_grid(N; ωmin=SB_OMEGA_MIN, ωmax=SB_OMEGA_MAX, C=SB_C)
    Δω = (ωmax - ωmin) / N
    frequencies = [ωmin + (k - 0.5) * Δω for k in 1:N]
    couplings = sqrt.(lorentzian_J.(frequencies; C) .* Δω)
    return frequencies, couplings
end

function thermal_boson_density(physical_site, liouville_site, ω, kBT, local_dim)
    occupations = 0:(local_dim - 1)
    if kBT <= 0
        weights = [n == 0 ? 1.0 : 0.0 for n in occupations]
    else
        weights = exp.(-ω .* occupations ./ kBT)
        weights ./= sum(weights)
    end
    number_states = [MPS([physical_site], [string(n)]) for n in occupations]
    density = to_dm(number_states; coeffs=weights)
    return to_liouville(density; sites=[liouville_site])
end

function lorentzian_spinboson_bath(kBT; n_modes, local_dim, C=SB_C)
    frequencies, couplings = lorentzian_mode_grid(n_modes; C)
    bath_sites = siteinds("Boson", n_modes; dim=local_dim)
    bath_liouville_sites = liouv_sites(bath_sites)
    modes = BosonicMode[]
    for k in 1:n_modes
        ωk = frequencies[k]
        gk = couplings[k]
        mode_hamiltonian = OpSum()
        mode_hamiltonian += ωk, "N", 1
        mode_coupling = OpSum()
        mode_coupling += gk, "A", 1, "ProjUp", 2
        mode_coupling += gk, "Adag", 1, "ProjUp", 2
        if abs(ωk) > 1e-12
            mode_coupling += gk^2 / ωk, "ProjUp", 2
        end
        push!(
            modes,
            bosonic_mode(
                [bath_liouville_sites[k]],
                mode_hamiltonian,
                thermal_boson_density(
                    bath_sites[k],
                    bath_liouville_sites[k],
                    ωk,
                    kBT,
                    local_dim,
                );
                coupling=mode_coupling,
            ),
        )
    end
    return with_logger(NullLogger()) do
        bosonic_bath(modes)
    end
end

function polarised_orientations(N)
    return [(; k, theta=0.0, phi=0.0, nx=0.0, ny=0.0, nz=1.0) for k in 1:N]
end

function driven_tls_system()
    sites = siteinds("S=1/2", 1)
    H = OpSum()
    H += SB_OMEGA, "Sx", 1
    system = with_logger(NullLogger()) do
        spin_system(sites, H)
    end
    return system
end

function ensure_csv_header!(path)
    if !isfile(path) || filesize(path) == 0
        write_csv(path, CSV_HEADER, [])
    end
    return path
end

function load_chi_baseline(path)
    baseline = Dict{Tuple{String,String},Int}()
    isfile(path) || return baseline
    lines = readlines(path)
    length(lines) < 2 && return baseline
    header = split(lines[1], ',')
    index = Dict(name => i for (i, name) in enumerate(header))
    for raw in lines[2:end]
        isempty(strip(raw)) && continue
        fields = split(raw, ',')
        key = (fields[index["case"]], fields[index["strategy"]])
        χ = parse(Int, fields[index["chi_max"]])
        haskey(baseline, key) || (baseline[key] = χ)
    end
    return baseline
end

function upsert_row!(path, row)
    lines = readlines(path)
    header = split(lines[1], ',')
    index = Dict(name => i for (i, name) in enumerate(header))
    case, model, strategy, nblas = string(row[1]), string(row[2]), string(row[3]), string(row[4])
    kept = String[lines[1]]
    for raw in lines[2:end]
        isempty(strip(raw)) && continue
        fields = split(raw, ',')
        same =
            fields[index["case"]] == case &&
            fields[index["strategy"]] == strategy &&
            fields[index["blas_threads"]] == nblas
        same || push!(kept, raw)
    end
    push!(kept, join(row, ","))
    open(path, "w") do io
        for line in kept
            println(io, line)
        end
    end
    return nothing
end

function cases_for_run()
    smoke = get(ENV, "ACE_THREAD_SMOKE", "0") == "1"
    easy_nsteps = smoke ? 4 : 200
    sb_te = parse(Float64, get(ENV, "ACE_THREAD_SB_TE", smoke ? "0.2" : "6.0"))
    sb_dt = 0.1
    sb_nsteps = max(1, round(Int, sb_te / sb_dt))
    sb_cutoff = parse(Float64, get(ENV, "ACE_THREAD_SB_CUTOFF", "1e-8"))
    sb_nmodes = parse(Int, get(ENV, "ACE_THREAD_SB_N", "8"))
    strategies = smoke ? (:zipup_cpp,) : STRATEGIES

    easy_system, _ = empty_spin_system()
    easy_bath = unpolarized_central_spin_bath(polarised_orientations(5))
    hard_system = driven_tls_system()
    hard_bath = lorentzian_spinboson_bath(3.0; n_modes=sb_nmodes, local_dim=5)

    return strategies, (
        (
            case="easy",
            model="central_spin_polarised",
            system=easy_system,
            bath=easy_bath,
            dt=0.1,
            nsteps=easy_nsteps,
            cutoff=1e-10,
        ),
        (
            case="hard",
            model="spinboson",
            system=hard_system,
            bath=hard_bath,
            dt=sb_dt,
            nsteps=sb_nsteps,
            cutoff=sb_cutoff,
        ),
    )
end

function main()
    nblas = parse(Int, get(ENV, "ACE_BLAS_THREADS", string(BLAS.get_num_threads())))
    nblas < 1 && throw(ArgumentError("ACE_BLAS_THREADS must be ≥ 1."))
    BLAS.set_num_threads(nblas)
    ensure_csv_header!(CSV_PATH)
    baseline = load_chi_baseline(CSV_PATH)
    strategies, cases = cases_for_run()

    println("ACE compressor BLAS-thread scaling")
    println("  Julia threads = $(Threads.nthreads())")
    println("  BLAS threads  = $(BLAS.get_num_threads())")
    println("  strategies    = $(join(strategies, ", "))")
    if Threads.nthreads() != 1
        println("WARNING: this experiment expects julia -t 1; only BLAS threads should vary.")
    end

    for spec in cases
        println("=== $(spec.case)  $(spec.model)  nsteps=$(spec.nsteps)  ε=$(spec.cutoff) ===")
        flush(stdout)
        for strategy in strategies
            @printf("  %-10s  warmup...\n", strategy)
            flush(stdout)
            warmup = @elapsed begin
                pt = build_ace_pt(
                    spec.system,
                    spec.bath;
                    cutoff=spec.cutoff,
                    compression=strategy,
                    dt=spec.dt,
                    nsteps=spec.nsteps,
                )
            end
            χ = Int(maxlinkdim(pt.core))
            @printf("  %-10s  warmup=%.1f s  χ=%d  timing...\n", strategy, warmup, χ)
            flush(stdout)
            stats = trial_stats(
                benchmark_ace_build(
                    spec.system,
                    spec.bath;
                    cutoff=spec.cutoff,
                    compression=strategy,
                    dt=spec.dt,
                    nsteps=spec.nsteps,
                ),
            )
            key = (spec.case, string(strategy))
            if haskey(baseline, key) && baseline[key] != χ
                @warn "χ changed with BLAS threads; do not treat this row as a scaling point" key χ expected=baseline[key] nblas
            else
                baseline[key] = χ
            end
            @printf(
                "  %-10s  t_min=%.3f  t_med=%.3f  χ=%d  BLAS=%d\n",
                strategy,
                stats.t_min_s,
                stats.t_median_s,
                χ,
                nblas,
            )
            flush(stdout)
            upsert_row!(
                CSV_PATH,
                (
                    spec.case,
                    spec.model,
                    strategy,
                    nblas,
                    Threads.nthreads(),
                    stats.t_min_s,
                    stats.t_median_s,
                    χ,
                    stats.memory_bytes,
                    stats.allocs,
                    stats.nsamples,
                ),
            )
        end
    end
    println("wrote $CSV_PATH")
    return nothing
end

main()
