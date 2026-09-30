# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# Provenance and CSV helpers for the Julia vs C++ ACE construction
# benchmarks. Physics parameters stay in common.jl.

function pin_compute_threads!()
    n = parse(Int, get(ENV, "JULIA_VS_CPP_BLAS_THREADS", "1"))
    n < 1 && throw(ArgumentError("BLAS thread count must be ≥ 1."))
    ENV["OMP_NUM_THREADS"] = string(n)
    ENV["MKL_NUM_THREADS"] = string(n)
    ENV["OPENBLAS_NUM_THREADS"] = string(n)
    ENV["MKL_DYNAMIC"] = "FALSE"
    ENV["OMP_DYNAMIC"] = "FALSE"
    BLAS.set_num_threads(n)
    return n
end

function package_version(mod::Module)
    try
        return string(pkgversion(mod))
    catch
        return "unknown"
    end
end

function package_version(name::AbstractString)
    name == "ProcessTensors" && return package_version(ProcessTensors)
    name == "ITensors" && return package_version(ITensors)
    try
        return package_version(getfield(Main, Symbol(name)))
    catch
    end
    deps = Pkg.dependencies()
    for (_, dep) in deps
        dep.name == name && return string(dep.version)
    end
    return "unknown"
end

function cpu_model()
    path = "/proc/cpuinfo"
    isfile(path) || return Sys.MACHINE
    for line in eachline(path)
        if startswith(line, "model name")
            return strip(split(line, ':'; limit=2)[2])
        end
    end
    return Sys.MACHINE
end

function available_ram()
    path = "/proc/meminfo"
    isfile(path) || return string(Sys.total_memory())
    for line in eachline(path)
        if startswith(line, "MemAvailable:") || startswith(line, "MemTotal:")
            kb = parse(Float64, split(line)[2])
            return @sprintf("%.1f GiB", kb / 1024^2)
        end
    end
    return string(Sys.total_memory())
end

function peak_rss_kb()
    path = "/proc/self/status"
    isfile(path) || return ""
    for line in eachline(path)
        if startswith(line, "VmHWM:")
            return split(line)[2]
        end
    end
    return ""
end

function swap_kb()
    path = "/proc/self/status"
    isfile(path) || return ""
    for line in eachline(path)
        if startswith(line, "VmSwap:")
            return split(line)[2]
        end
    end
    return ""
end

function pinned_cpu()
    get(ENV, "JULIA_VS_CPP_PIN_CPU", "")
end

function requested_cpus()
    raw = get(ENV, "JULIA_VS_CPP_PIN_CPU", "")
    isempty(strip(raw)) && (raw = get(ENV, "JULIA_VS_CPP_CPUS", get(ENV, "JULIA_SPINBOSON_CPUS", "")))
    return [parse(Int, tok) for tok in split(replace(strip(raw), ',' => ' ')) if !isempty(tok)]
end

function cpu_is_busy(cpu::Int)
    my = getpid()
    for name in readdir("/proc"; sort=false)
        all(isdigit, name) || continue
        parse(Int, name) == my && continue
        try
            stat = read("/proc/$name/stat", String)
            rpar = findlast(')', stat)
            rpar === nothing && continue
            fields = split(SubString(stat, rpar + 2))
            length(fields) < 37 && continue
            fields[1] == "R" || continue
            parse(Int, fields[37]) == cpu && return true
        catch
        end
    end
    return false
end

function pin_to_cpu!(cpu::Int)
    mask = zeros(UInt8, 128)
    cpu < 8 * length(mask) || throw(ArgumentError("CPU $cpu is out of range"))
    mask[cpu ÷ 8 + 1] |= UInt8(1) << (cpu % 8)
    ccall(:sched_setaffinity, Cint, (Cint, Csize_t, Ptr{UInt8}), 0, sizeof(mask), mask) == 0 ||
        error("Could not pin to CPU $cpu")
    ENV["JULIA_VS_CPP_PIN_CPU"] = string(cpu)
    return cpu
end

function pin_next_task_cpu!(case_index::Integer)
    cpus = requested_cpus()
    isempty(cpus) && return nothing
    preferred = cpus[mod1(case_index, length(cpus))]
    n = max(Sys.CPU_THREADS, preferred + 1)
    cpu = preferred
    for _ in 1:n
        if !cpu_is_busy(cpu)
            cpu == preferred || println(
                "CPU $preferred is under use, so we moved this task to CPU $cpu that is free",
            )
            return pin_to_cpu!(cpu)
        end
        cpu = mod(cpu + 1, n)
    end
    error("No free CPU found (preferred $preferred)")
end


function cpus_allowed()
    path = "/proc/self/status"
    isfile(path) || return pinned_cpu()
    for line in eachline(path)
        if startswith(line, "Cpus_allowed_list:")
            return strip(split(line, ':'; limit=2)[2])
        end
    end
    return pinned_cpu()
end

function print_timing_policy()
    println("""
    TIMING POLICY
      This is the Julia–C++ construction comparison. The runner times
      build_process_tensor only (setup is recorded separately). Compare
      C++ construction with build_s. Table S.3.1 complete-example times
      live only in the sanity folders.
    """)
    return nothing
end

function print_julia_provenance()
    println("ACE JULIA BENCHMARK PROVENANCE")
    println("Julia version        : $VERSION")
    println("ProcessTensors       : $(package_version("ProcessTensors"))")
    println("ITensors             : $(package_version("ITensors"))")
    println("ITensorMPS           : $(package_version("ITensorMPS"))")
    println("BLAS                 : $(first(split(string(BLAS.get_config()), '\n')))")
    println("Julia threads        : $(Threads.nthreads())")
    println("BLAS threads         : $(BLAS.get_num_threads())")
    println("OMP threads          : $(get(ENV, "OMP_NUM_THREADS", "unset"))")
    println("MKL threads          : $(get(ENV, "MKL_NUM_THREADS", "unset"))")
    println("OPENBLAS threads     : $(get(ENV, "OPENBLAS_NUM_THREADS", "unset"))")
    println("CPU model            : $(cpu_model())")
    println("logical CPUs         : $(Sys.CPU_THREADS)")
    println("available RAM        : $(available_ram())")
    println("host/kernel          : $(Sys.KERNEL) $(Sys.MACHINE)")
    println("cpus allowed         : $(cpus_allowed())")
    println("timing clock         : BenchmarkTools nanosecond wall time")
    println("timing scope         : setup / build / contraction / I/O / total")
    println()
    return nothing
end

function print_run_parameters(;
    run_id,
    model,
    bath_state,
    N_modes,
    local_dim,
    polarization,
    dt,
    t_final,
    nsteps,
    cutoff,
    symmetric_trotter="Trotter{2}()",
    seed="",
    temperature="",
    spectral_parameters="",
    compression="",
)
    println("RUN PARAMETERS")
    println("run_id               : $run_id")
    println("model                : $model")
    println("bath_state           : $bath_state")
    println("N_modes              : $N_modes")
    println("local_dim            : $local_dim")
    println("polarization         : $polarization")
    println("dt                   : $dt")
    println("t_final              : $t_final")
    println("nsteps               : $nsteps")
    println("cutoff               : $cutoff")
    println("symmetric_trotter    : $symmetric_trotter")
    println("seed                 : $seed")
    println("temperature          : $temperature")
    println("spectral_parameters  : $spectral_parameters")
    println("compression          : $compression")
    println("pinned_cpu           : $(cpus_allowed())")
    println("Julia/BLAS threads   : $(Threads.nthreads())/$(BLAS.get_num_threads())")
    println()
    return nothing
end

function print_results_block(;
    maxdim,
    setup_s,
    build_s,
    contract_s,
    io_s,
    total_s,
    peak_rss_kb="",
    swap_io="",
    timing_scope="build-only",
)
    println("RESULTS")
    println("D_max                : $maxdim")
    println("setup_s              : $setup_s")
    println("build_s              : $build_s")
    println("contract_s           : $contract_s")
    println("io_s                 : $io_s")
    println("total_s              : $total_s")
    println("peak_rss_kb          : $peak_rss_kb")
    println("swap_io              : $swap_io")
    println("timing_scope         : $timing_scope")
    println()
    return nothing
end

function print_validation(;
    maxdim,
    final_trace_error="n/a (build-only)",
    trajectory_checksum="n/a (build-only)",
    max_reference_error="n/a (build-only)",
    status,
)
    println("VALIDATION")
    println("max_bond_dimension   : $maxdim")
    println("final_trace_error    : $final_trace_error")
    println("trajectory_checksum  : $trajectory_checksum")
    println("max_reference_error  : $max_reference_error")
    println("status               : $status")
    println()
    return nothing
end

function csv_escape(value)
    text = string(value)
    return occursin(r"[,\"\n]", text) ? '"' * replace(text, '"' => "\"\"") * '"' : text
end

function ensure_csv_header!(path::AbstractString, header::AbstractString)
    columns = split(header, ',')
    if !isfile(path) || filesize(path) == 0
        mkpath(dirname(path))
        open(path, "w") do io
            println(io, header)
        end
        return :created
    end
    lines = readlines(path)
    old_header = split(lines[1], ',')
    old_header == columns && return :unchanged
    index = Dict(name => i for (i, name) in enumerate(old_header))
    open(path, "w") do io
        println(io, header)
        for raw in lines[2:end]
            isempty(strip(raw)) && continue
            fields = split(raw, ',')
            mapped = map(columns) do name
                i = get(index, name, 0)
                1 <= i <= length(fields) ? fields[i] : ""
            end
            values = Dict(columns[i] => mapped[i] for i in eachindex(columns))
            elapsed = values["elapsed_sec"]
            isempty(elapsed) && haskey(values, "t_min_s") && (elapsed = values["t_min_s"])
            for name in ("build_s", "total_s")
                haskey(values, name) && isempty(values[name]) && !isempty(elapsed) && (values[name] = elapsed)
            end
            for name in ("contract_s", "io_s")
                haskey(values, name) && isempty(values[name]) && (values[name] = "0")
            end
            println(io, join([values[name] for name in columns], ','))
        end
    end
    return :upgraded
end

const CENTRAL_SPIN_CSV_HEADER = join(
    [
        "code,model,polarisation,N,Jk,dt,te,threshold,t_min_s,t_median_s,t_mean_s,maxdim,nsamples,memory_bytes,allocs,nthreads,blas_threads,compression",
        "setup_s,build_s,contract_s,io_s,total_s,peak_rss_kb,swap_io,timing_scope,omp_threads,mkl_threads,pinned_cpu,max_bond_dimension,final_trace_error,trajectory_checksum,max_reference_error,validation_status",
    ],
    ",",
)

const SPINBOSON_CSV_HEADER = join(
    [
        "code,model,kBT_over_Omega,N_modes,M,C_over_Omega2,dt,te,threshold,elapsed_sec,maxdim,compression",
        "setup_s,build_s,contract_s,io_s,total_s,peak_rss_kb,swap_io,timing_scope,nthreads,blas_threads,omp_threads,mkl_threads,pinned_cpu,max_bond_dimension,final_trace_error,trajectory_checksum,max_reference_error,validation_status",
    ],
    ",",
)

function csv_has_case(csv::AbstractString, test)
    isfile(csv) || return false
    lines = readlines(csv)
    length(lines) < 2 && return false
    header = split(lines[1], ',')
    index = Dict(name => i for (i, name) in enumerate(header))
    for raw in lines[2:end]
        isempty(strip(raw)) && continue
        fields = split(raw, ',')
        test(header, index, fields) && return true
    end
    return false
end
