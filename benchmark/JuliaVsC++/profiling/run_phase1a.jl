# Phase 1A. Uninstrumented reference, then instrumented gesvd and gesdd builds.
# Partial central spin, N=5, the existing benchmark parameters.

include(joinpath(@__DIR__, "..", "common.jl"))
import ITensorMPS
include(joinpath(@__DIR__, "probe.jl"))

const PROFILE_DIR = @__DIR__
const PROFILE_RESULTS = joinpath(PROFILE_DIR, "results")
const PROFILE_LOGS = joinpath(PROFILE_DIR, "logs")
const PROFILE_FIXTURES = joinpath(PROFILE_DIR, "fixtures")

function blas_lib_name()
    cfg = BLAS.get_config()
    libs = cfg.loaded_libs
    isempty(libs) && return "unknown"
    return libs[1].libname
end

function write_provenance(path)
    open(path, "w") do io
        println(io, "julia $(VERSION)")
        println(io, "ProcessTensors $(pkgversion(ProcessTensors))")
        println(io, "ITensors $(pkgversion(ITensors))")
        println(io, "ITensorMPS $(pkgversion(ITensorMPS))")
        println(io, "blas $(blas_lib_name())")
        println(io, "blas_config $(BLAS.get_config())")
        println(io, "blas_int_bits $(8 * sizeof(LinearAlgebra.BlasInt))")
        println(io, "julia_threads $(Threads.nthreads())")
        println(io, "blas_threads $(BLAS.get_num_threads())")
        println(io, "omp $(get(ENV, "OMP_NUM_THREADS", ""))")
        println(io, "openblas_env $(get(ENV, "OPENBLAS_NUM_THREADS", ""))")
        println(io, "cpu $(get(ENV, "PROFILING_CPU", cpus_allowed()))")
        println(io, "probe_lib $(get(ENV, "SVD_COUNT_LIB", ""))")
        println(io, "hostname $(gethostname())")
    end
    return path
end

function open_log(path, dump_ids)
    mkpath(dirname(path))
    io = open(path, "w")
    println(io, PER_SVD_HEADER)
    return ProbeLog(io, 0, dump_ids, PROFILE_FIXTURES, 0.0, 0)
end

function summarise(path, log, build_s, maxdim, solver)
    rows = readlines(path)[2:end]
    filter!(!isempty, rows)
    cols = split(PER_SVD_HEADER, ",")
    idx = Dict(name => i for (i, name) in enumerate(cols))
    function total(name)
        j = idx[name]
        return sum(parse(Float64, split(row, ",")[j]) for row in rows)
    end
    open(joinpath(PROFILE_RESULTS, "stage_times.csv"), "a") do io
        println(
            io,
            join((
                solver, build_s, length(rows), maxdim,
                log.contraction_s, total("combine_permute_s"), total("matricise_s"),
                total("lapack_s"), total("truncate_s"), total("reassemble_s"),
                max(build_s - log.contraction_s - total("combine_permute_s") - total("matricise_s") - total("lapack_s") - total("truncate_s") - total("reassemble_s"), 0.0),
                log.contraction_bytes,
            ), ","),
        )
    end
    return length(rows)
end

function run_instrumented(system, bath, nsteps, svd_alg, csv_name, dump_ids)
    solver = svd_alg === nothing ? "divide_and_conquer" : String(svd_alg)
    path = joinpath(PROFILE_RESULTS, csv_name)
    log = open_log(path, dump_ids)
    build_s = @elapsed cores = instrumented_central_spin(
        system, bath;
        dt=CS_DT, nsteps, cutoff=CS_CUTOFF, maxdim=CS_MAXDIM, svd_alg, log,
    )
    close(log.io)
    χ = Int(ITensorMPS.maxlinkdim(ITensorMPS.MPO(cores)))
    χ == 65 || error("$solver instrumented build produced D_max=$χ, expected 65")
    if get(ENV, "PROBE_RECORD_STAGES", "1") == "1"
        summarise(path, log, build_s, χ, solver)
    end
    println("instrumented solver=$solver build_s=$build_s D_max=$χ rows=$(log.call)")
    return build_s, χ
end

function main()
    pin_compute_threads!()
    Threads.nthreads() == 1 && BLAS.get_num_threads() == 1 || error(
        "profiling requires julia -t 1 and one BLAS thread",
    )
    mkpath(PROFILE_RESULTS)
    mkpath(PROFILE_LOGS)
    mkpath(PROFILE_FIXTURES)
    lib = get(ENV, "SVD_COUNT_LIB", joinpath(PROFILE_DIR, "cpp", "libsvd_count.so"))
    isfile(lib) || error("missing $lib; build profiling/cpp first")
    PROBE_LIB[] = lib
    write_provenance(joinpath(PROFILE_RESULTS, "provenance.txt"))

    if get(ENV, "PROBE_SKIP_SELFCHECK", "0") != "1"
        println("self-check against ITensors.svd")
        meta = selfcheck_probe()
        println("self-check ok  kernel_calls=$(meta.kernel_calls) job=$(meta.jobu)$(meta.jobvt) kind=$(meta.kind)")
    end

    nsteps = nsteps_for(CS_FINAL_TIME, CS_DT)
    system = empty_central_spin_system()
    orientations = orientations_for("partial", 5)
    bath = central_spin_bath(orientations)

    if get(ENV, "PROBE_SKIP_REFERENCE", "0") == "1"
        println("skipping uninstrumented reference")
        dump_ids = Set{Int}()
        raw = get(ENV, "PROBE_DUMP_IDS", "")
        if !isempty(raw)
            dump_ids = Set(parse(Int, strip(s)) for s in split(raw, ",") if !isempty(strip(s)))
        end
        run_instrumented(system, bath, nsteps, "qr_iteration", "per_svd_dump.csv", dump_ids)
        println("dump pass wrote fixtures")
        return nothing
    end

    println("uninstrumented gesvd reference")
    warmup_s = @elapsed build_ace_pt(
        system, bath; dt=CS_DT, nsteps, cutoff=CS_CUTOFF, maxdim=CS_MAXDIM,
        compression=:zipup_cpp, svd_alg="qr_iteration",
    )
    build_s = @elapsed pt = build_ace_pt(
        system, bath; dt=CS_DT, nsteps, cutoff=CS_CUTOFF, maxdim=CS_MAXDIM,
        compression=:zipup_cpp, svd_alg="qr_iteration",
    )
    χ = Int(maxlinkdim(pt.core))
    χ == 65 || error("reference gesvd D_max=$χ, expected 65")
    open(joinpath(PROFILE_RESULTS, "reference.csv"), "w") do io
        println(io, "solver,warmup_s,build_s,maxdim")
        println(io, "qr_iteration,$warmup_s,$build_s,$χ")
    end
    println("reference gesvd warmup_s=$warmup_s build_s=$build_s D_max=$χ")

    open(joinpath(PROFILE_RESULTS, "stage_times.csv"), "w") do io
        println(io, "solver,build_s,n_svd,maxdim,contraction_s,combine_permute_s,matricise_s,lapack_s,truncate_s,reassemble_s,other_s,contraction_bytes")
    end
    dump_ids = Set{Int}()
    raw = get(ENV, "PROBE_DUMP_IDS", "")
    if !isempty(raw)
        dump_ids = Set(parse(Int, strip(s)) for s in split(raw, ",") if !isempty(strip(s)))
    end
    only = get(ENV, "PROBE_ONLY", "")
    if only == "" || only == "gesvd"
        run_instrumented(system, bath, nsteps, "qr_iteration", "per_svd_gesvd.csv", only == "gesvd" ? dump_ids : Set{Int}())
    end
    if isempty(dump_ids) && (only == "" || only == "gesdd")
        run_instrumented(system, bath, nsteps, nothing, "per_svd_gesdd.csv", Set{Int}())
    end
    println("phase 1A wrote $PROFILE_RESULTS")
    return nothing
end

main()
