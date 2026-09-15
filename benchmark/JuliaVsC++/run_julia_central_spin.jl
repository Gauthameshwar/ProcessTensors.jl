# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# Time ProcessTensors.jl ACE construction for the three central-spin
# polarisations and N in {5,10,25,50,100}.
#
# Compilation / first-use JIT is discarded: each case does one untimed
# warmup `build_process_tensor`, then BenchmarkTools times construction only
# (`evals=1`). Bath / orientation setup is outside the timed region.
#
# Single-core head-to-head (default):
#
#   JULIA_VS_CPP_CPUS="8 9 10 11" \
#     OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
#     julia -t 1 --project=. benchmark/JuliaVsC++/run_julia_central_spin.jl
#
# Smoke (polarised N=5 only):
#
#   JULIA_VS_CPP_SMOKE=1 julia -t 1 --project=. \
#     benchmark/JuliaVsC++/run_julia_central_spin.jl
#
# Writes only `results/julia/central_spin.csv`. C++ artefacts are read, never
# overwritten. Physics parameters are unchanged.

include(joinpath(@__DIR__, "common.jl"))

const CENTRAL_SPIN_HEADER = CENTRAL_SPIN_CSV_HEADER

function selected_cases()
    if get(ENV, "JULIA_VS_CPP_SMOKE", "0") == "1"
        return [("polarised", 5)]
    end
    pols = split(get(ENV, "JULIA_VS_CPP_POLS", "polarised,partial,unpolarised"), ',')
    ns = [parse(Int, s) for s in split(get(ENV, "JULIA_VS_CPP_NS", "5,10,25,50,100"), ',')]
    return [(strip(p), N) for p in pols for N in ns]
end

polarization_label(polarisation) =
    polarisation == "polarised" ? "inf (+z)" :
    polarisation == "partial" ? "b=20" :
    polarisation == "unpolarised" ? "b=0" : polarisation

function main()
    ensure_results_dir()
    pin_compute_threads!()
    csv = joinpath(
        RESULTS_DIR,
        get(ENV, "JULIA_VS_CPP_OUTPUT", "central_spin.csv"),
    )
    nsteps = nsteps_for(CS_FINAL_TIME, CS_DT)
    system = empty_central_spin_system()
    cases = selected_cases()
    J = parse(Float64, get(ENV, "JULIA_VS_CPP_J", string(CS_J)))
    compression = Symbol(get(ENV, "JULIA_VS_CPP_COMPRESSION", "zipup_cpp"))
    compression in (:zipup_cpp, :canonzip) || throw(
        ArgumentError("Unsupported JULIA_VS_CPP_COMPRESSION=$compression"),
    )

    print_julia_provenance()
    print_timing_policy()
    println("BenchmarkTools samples=$(BENCH_SAMPLES)  seconds=$(BENCH_SECONDS)")
    println("Cases: $cases")
    if Threads.nthreads() != 1 || BLAS.get_num_threads() != 1
        println("WARNING: head-to-head comparison expects julia -t 1 and BLAS threads=1.")
    end
    println()

    open(csv, "w") do io
        println(io, CENTRAL_SPIN_HEADER)
        for (case_index, (polarisation, N)) in enumerate(cases)
            pin_next_task_cpu!(case_index)

            Jk = J / N
            run_id = "julia_central_$(polarisation)_N$(N)"
            print_run_parameters(;
                run_id,
                model="central_spin",
                bath_state=polarisation,
                N_modes=N,
                local_dim=2,
                polarization=polarization_label(polarisation),
                dt=CS_DT,
                t_final=CS_FINAL_TIME,
                nsteps,
                cutoff=CS_CUTOFF,
                seed=polarisation == "polarised" ? "n/a (all +z)" : "C++ orientations dump",
                compression=string(compression),
            )
            setup = @elapsed begin
                orientations = orientations_for(polarisation, N)
                bath = central_spin_bath(orientations; J)
            end
            Sx, Sy, Sz = total_environment_spin(orientations)
            println("=== Julia ACE  polarisation=$polarisation  N=$N  J_k=$Jk ===")
            @printf("  bath  <S> = (%.6f, %.6f, %.6f)\n", Sx, Sy, Sz)
            _, stats, χ = measure_ace_pt(
                system,
                bath;
                dt=CS_DT,
                nsteps,
                cutoff=CS_CUTOFF,
                maxdim=CS_MAXDIM,
                compression,
            )
            build_s = stats.t_min_s
            contract_s = 0.0
            io_s = 0.0
            total_s = setup + build_s
            rss = peak_rss_kb()
            swap = swap_kb()
            status = χ >= 1 ? "PASS" : "FAIL"
            print_results_block(;
                maxdim=χ,
                setup_s=setup,
                build_s,
                contract_s,
                io_s,
                total_s,
                peak_rss_kb=rss,
                swap_io=swap,
                timing_scope="build-only (BenchmarkTools t_min after warmup)",
            )
            print_validation(; maxdim=χ, status)
            @printf(
                "  t_min=%.6f  t_median=%.6f  maxdim=%d  nsamples=%d\n",
                stats.t_min_s,
                stats.t_median_s,
                χ,
                stats.nsamples,
            )
            println(
                io,
                join(
                    [
                        "julia",
                        "central_spin",
                        polarisation,
                        N,
                        Jk,
                        CS_DT,
                        CS_FINAL_TIME,
                        CS_CUTOFF,
                        stats.t_min_s,
                        stats.t_median_s,
                        stats.t_mean_s,
                        χ,
                        stats.nsamples,
                        stats.memory_bytes,
                        stats.allocs,
                        Threads.nthreads(),
                        BLAS.get_num_threads(),
                        compression,
                        setup,
                        build_s,
                        contract_s,
                        io_s,
                        total_s,
                        rss,
                        swap,
                        "build-only",
                        get(ENV, "OMP_NUM_THREADS", ""),
                        get(ENV, "MKL_NUM_THREADS", ""),
                        cpus_allowed(),
                        χ,
                        "n/a",
                        "n/a",
                        "n/a",
                        status,
                    ],
                    ",",
                ),
            )
            flush(io)
        end
    end
    println("wrote $csv")
    return nothing
end

main()
