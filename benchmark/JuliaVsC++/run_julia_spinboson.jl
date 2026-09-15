# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# Time ProcessTensors.jl ACE construction for the Lorentzian spin-boson bath.
# Skip a CSV row only when N_E, cutoff, compressor, and thread counts all
# match; blank thread fields are treated as a miss and the point is rerun.
#
#   bash benchmark/JuliaVsC++/run_julia_spinboson_parallel.sh
#
# Serial:
#   OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
#     julia -t 1 --project=. benchmark/JuliaVsC++/run_julia_spinboson.jl

include(joinpath(@__DIR__, "common.jl"))

function selected_spinboson_cases()
    temperatures = parse.(Float64, split(get(ENV, "JULIA_VS_CPP_TEMPS", join(SB_TEMPERATURES, ',')), ','))
    mode_counts = parse.(Int, split(get(ENV, "JULIA_VS_CPP_NS", join(SB_NS, ',')), ','))
    return [(temperature, n_modes) for temperature in temperatures for n_modes in mode_counts]
end

function csv_val(index, fields, name)
    i = get(index, name, 0)
    return 1 <= i <= length(fields) ? strip(fields[i]) : ""
end

function spinboson_case_done(csv, kBT, n_modes; cutoff, compression, threads)
    isfile(csv) || return false
    lines = readlines(csv)
    length(lines) < 2 && return false
    header = split(lines[1], ',')
    index = Dict(name => i for (i, name) in enumerate(header))
    want = lowercase(string(compression))
    for raw in lines[2:end]
        isempty(strip(raw)) && continue
        fields = split(strip(raw), ',')
        csv_val(index, fields, "model") == "spinboson" || continue
        try
            parse(Float64, csv_val(index, fields, "kBT_over_Omega")) ≈ kBT || continue
            parse(Int, csv_val(index, fields, "N_modes")) == n_modes || continue
            parse(Float64, csv_val(index, fields, "threshold")) ≈ Float64(cutoff) || continue
        catch
            continue
        end
        lowercase(lstrip(csv_val(index, fields, "compression"), ':')) == want || continue
        thread_s = csv_val(index, fields, "blas_threads")
        isempty(thread_s) && (thread_s = csv_val(index, fields, "nthreads"))
        isempty(thread_s) && (thread_s = csv_val(index, fields, "omp_threads"))
        isempty(thread_s) && continue
        try
            parse(Int, thread_s) == threads || continue
        catch
            continue
        end
        return true
    end
    return false
end

function main()
    ensure_results_dir()
    pin_compute_threads!()
    csv = joinpath(RESULTS_DIR, get(ENV, "JULIA_VS_CPP_OUTPUT", "spinboson.csv"))
    nsteps = nsteps_for(SB_FINAL_TIME, SB_DT)
    system = driven_tls_system()
    compression = Symbol(get(ENV, "JULIA_VS_CPP_COMPRESSION", "zipup_cpp"))
    compression === :zipup_cpp || throw(
        ArgumentError("Spin-boson head-to-head requires JULIA_VS_CPP_COMPRESSION=zipup_cpp; got $compression."),
    )
    (Threads.nthreads() == 1 && BLAS.get_num_threads() == 1) || error(
        "Spin-boson head-to-head requires julia -t 1 and BLAS threads=1; got Julia=$(Threads.nthreads()) BLAS=$(BLAS.get_num_threads()).",
    )
    ensure_csv_header!(csv, SPINBOSON_CSV_HEADER)
    cases = selected_spinboson_cases()

    if get(ENV, "JULIA_VS_CPP_PRINT_PROVENANCE", "1") != "0"
        print_julia_provenance()
        print_timing_policy()
    end
    println("Compression = $compression")
    println("Cases = $cases")
    println()

    open(csv, "a") do io
        task = 0
        for (kBT, n_modes) in cases
            if spinboson_case_done(
                csv, kBT, n_modes;
                cutoff=SB_CUTOFF, compression, threads=1,
            )
                println("Skipping completed Julia case kBT/Omega=$kBT N_modes=$n_modes (zipup_cpp, ε=$SB_CUTOFF, threads=1)")
                continue
            end
            task += 1
            pin_next_task_cpu!(task)
            cpu = cpus_allowed()
            println("CPU $cpu  <-  kBT/Omega=$kBT  N_modes=$n_modes  compression=$compression  BLAS=$(BLAS.get_num_threads())")

            run_id = "julia_spinboson_T$(kBT)_N$(n_modes)"
            print_run_parameters(;
                run_id,
                model="spinboson",
                bath_state="thermal",
                N_modes=n_modes,
                local_dim=SB_LOCAL_DIM,
                polarization="n/a",
                dt=SB_DT,
                t_final=SB_FINAL_TIME,
                nsteps,
                cutoff=SB_CUTOFF,
                temperature=kBT,
                spectral_parameters="Lorentzian C=$(SB_C) gamma=$(SB_GAMMA) omega_c=$(SB_OMEGA_C) omega in [$(SB_OMEGA_MIN),$(SB_OMEGA_MAX)]",
                compression=string(compression),
            )
            println("=== Julia ACE  spin-boson  kBT/Omega=$kBT  N_modes=$n_modes  cpu=$cpu ===")
            setup = @elapsed begin
                bath = lorentzian_spinboson_bath(kBT; n_modes)
            end
            _, stats, χ = measure_ace_pt(
                system, bath;
                dt=SB_DT, nsteps, cutoff=SB_CUTOFF, maxdim=SB_MAXDIM, compression,
            )
            build_s = stats.t_min_s
            status = χ >= 1 ? "PASS" : "FAIL"
            print_results_block(;
                maxdim=χ,
                setup_s=setup,
                build_s,
                contract_s=0.0,
                io_s=0.0,
                total_s=setup + build_s,
                peak_rss_kb=peak_rss_kb(),
                swap_io=swap_kb(),
                timing_scope="build-only (BenchmarkTools t_min after warmup)",
            )
            print_validation(; maxdim=χ, status)
            @printf("  t_min=%.6f  t_median=%.6f  maxdim=%d\n", stats.t_min_s, stats.t_median_s, χ)
            println(
                io,
                join(
                    [
                        "julia", "spinboson", kBT, n_modes, SB_LOCAL_DIM, SB_C, SB_DT,
                        SB_FINAL_TIME, SB_CUTOFF, stats.t_min_s, χ, compression, setup,
                        build_s, 0.0, 0.0, setup + build_s, peak_rss_kb(), swap_kb(),
                        "build-only", Threads.nthreads(), BLAS.get_num_threads(),
                        get(ENV, "OMP_NUM_THREADS", ""), get(ENV, "MKL_NUM_THREADS", ""),
                        cpu, χ, "n/a", "n/a", "n/a", status,
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
