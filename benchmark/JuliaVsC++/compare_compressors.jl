# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# Time the two Julia ACE schedules on the same Julia-vs-C++ central-spin
# process tensors, then sit them next to the C++ constructor.
#
#   OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
#     julia -t 1 --project=. benchmark/JuliaVsC++/compare_compressors.jl
#
# Optional:
#   JULIA_VS_CPP_CASES="polarised:5,polarised:10" ACE_BENCH_SAMPLES=1

include(joinpath(@__DIR__, "common.jl"))

const STRATEGIES = (:zipup_cpp, :canonzip)
const OUT_CSV = joinpath(RESULTS_DIR, "compressor_side_by_side.csv")

function parse_cases()
    raw = get(ENV, "JULIA_VS_CPP_CASES", "polarised:5,polarised:10,polarised:25,unpolarised:5")
    cases = Tuple{String,Int}[]
    for token in split(raw, ',')
        pol, n = split(strip(token), ':')
        push!(cases, (String(pol), parse(Int, n)))
    end
    return cases
end

function load_named_csv(path)
    isfile(path) || return Dict{Tuple{String,Int},NamedTuple}()
    lines = readlines(path)
    isempty(lines) && return Dict{Tuple{String,Int},NamedTuple}()
    header = split(lines[1], ',')
    index = Dict(name => i for (i, name) in enumerate(header))
    haskey(index, "polarisation") || return Dict{Tuple{String,Int},NamedTuple}()
    tcol = haskey(index, "elapsed_sec") ? "elapsed_sec" :
        haskey(index, "t_min_s") ? "t_min_s" :
        haskey(index, "build_s") ? "build_s" : nothing
    tcol === nothing && return Dict{Tuple{String,Int},NamedTuple}()
    table = Dict{Tuple{String,Int},NamedTuple}()
    for raw in lines[2:end]
        isempty(strip(raw)) && continue
        fields = split(raw, ',')
        pol = fields[index["polarisation"]]
        N = parse(Int, fields[index["N"]])
        t = parse(Float64, fields[index[tcol]])
        χ = parse(Int, fields[index["maxdim"]])
        table[(pol, N)] = (t=t, maxdim=χ)
    end
    return table
end

function main()
    ensure_results_dir()
    pin_compute_threads!()
    print_julia_provenance()
    print_timing_policy()

    cases = parse_cases()
    nsteps = nsteps_for(CS_FINAL_TIME, CS_DT)
    system = empty_central_spin_system()
    cpp_now = load_named_csv(joinpath(CPP_RESULTS_DIR, "central_spin.csv"))
    cpp_old = load_named_csv(joinpath(@__DIR__, "results", "archive_old", "central_spin.csv"))
    julia_now = load_named_csv(joinpath(RESULTS_DIR, "central_spin.csv"))

    println("SIDE-BY-SIDE COMPRESSORS")
    println("  model     : central_spin  dt=$(CS_DT)  te=$(CS_FINAL_TIME)  ε=$(CS_CUTOFF)")
    println("  nsteps    : $nsteps")
    println("  cases     : $cases")
    println("  strategies: $(join(STRATEGIES, ", "))")
    println()

    header = [
        "code",
        "polarisation",
        "N",
        "compression",
        "t_min_s",
        "maxdim",
        "memory_bytes",
        "allocs",
        "nthreads",
        "blas_threads",
    ]
    rows = Any[]

    open(OUT_CSV, "w") do io
        println(io, join(header, ","))
        for (polarisation, N) in cases
            orientations = orientations_for(polarisation, N)
            bath = central_spin_bath(orientations; J=CS_J)
            println("=== $polarisation  N=$N ===")
            for compression in STRATEGIES
                _, stats, χ = measure_ace_pt(
                    system,
                    bath;
                    dt=CS_DT,
                    nsteps,
                    cutoff=CS_CUTOFF,
                    maxdim=CS_MAXDIM,
                    compression,
                )
                @printf(
                    "  %-10s  t_min=%.3f s  χ=%d  allocs=%d\n",
                    compression,
                    stats.t_min_s,
                    χ,
                    stats.allocs,
                )
                row = [
                    "julia",
                    polarisation,
                    N,
                    compression,
                    stats.t_min_s,
                    χ,
                    stats.memory_bytes,
                    stats.allocs,
                    Threads.nthreads(),
                    BLAS.get_num_threads(),
                ]
                println(io, join(row, ","))
                push!(rows, (polarisation, N, compression, stats.t_min_s, χ))
            end
            println()
        end
    end

    grouped = Dict{Tuple{String,Int},Dict{Symbol,NamedTuple}}()
    for (pol, N, compression, t, χ) in rows
        get!(grouped, (pol, N), Dict{Symbol,NamedTuple}())
        grouped[(pol, N)][compression] = (t=t, maxdim=χ)
    end

    println("COMPARISON TABLE")
    println(
        rpad("case", 22),
        rpad("zipup_cpp", 16),
        rpad("canonzip", 16),
        rpad("C++ now", 16),
        rpad("C++ old", 16),
        "Julia CSV",
    )
    for (pol, N) in cases
        g = grouped[(pol, N)]
        function cell(src)
            src === nothing && return rpad("—", 16)
            return rpad(@sprintf("%.2f / %d", src.t, src.maxdim), 16)
        end
        now_j = get(julia_now, (pol, N), nothing)
        print(rpad("$pol N=$N", 22))
        print(cell(g[:zipup_cpp]))
        print(cell(g[:canonzip]))
        print(cell(get(cpp_now, (pol, N), nothing)))
        print(cell(get(cpp_old, (pol, N), nothing)))
        println(now_j === nothing ? "—" : @sprintf("%.2f / %d", now_j.t, now_j.maxdim))
    end
    println()
    println("Cells are  t_min_s / maxdim. C++ times are ACE wall (dont_propagate).")
    println("Julia CSV: current single-core results/julia/central_spin.csv.")
    println("wrote $OUT_CSV")

    scaling = joinpath(@__DIR__, "..", "ace_compressors", "results", "ace_central_spin_scaling.csv")
    if isfile(scaling)
        println()
        println("EXISTING ace_compressors scaling (different model: unpolarised, dt=0.05, T=4)")
        println(read(scaling, String))
    end
    return nothing
end

main()
