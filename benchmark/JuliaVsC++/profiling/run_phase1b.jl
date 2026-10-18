# Phase 1B. Replay the small, median-cost, and most expensive gesvd matrices.

include(joinpath(@__DIR__, "..", "common.jl"))

import ITensors
using BenchmarkTools
using DelimitedFiles
using LinearAlgebra

const PROFILE_DIR = @__DIR__
const PROFILE_RESULTS = joinpath(PROFILE_DIR, "results")
const PROFILE_FIXTURES = joinpath(PROFILE_DIR, "fixtures")

function load_trace()
    path = joinpath(PROFILE_RESULTS, "per_svd_gesvd.csv")
    isfile(path) || error("missing $path")
    rows, header = readdlm(path, ','; header=true)
    names = string.(vec(header))
    return [Dict(names[j] => rows[i, j] for j in eachindex(names)) for i in axes(rows, 1)]
end

function pick_calls(rows)
    costs = [(Int(row["call_index"]), Float64(row["cost"]), Int(row["m"]), Int(row["n"])) for row in rows]
    filter!(item -> item[2] > 0, costs)
    isempty(costs) && error("no positive-cost SVDs")
    small = argmin(item -> (item[3] * item[4], item[1]), costs)
    expensive = argmax(item -> (item[2], -item[1]), costs)
    ordered = sort(costs; by=item -> (item[2], item[1]))
    median = ordered[cld(length(ordered), 2)]
    labels = Dict(small[1] => "small", median[1] => "median", expensive[1] => "expensive")
    return labels
end

function read_matrix(path)
    open(path, "r") do io
        m = read(io, Int64)
        n = read(io, Int64)
        data = Array{ComplexF64}(undef, m, n)
        read!(io, data)
        return data
    end
end

function ace_rank(sigma, epsilon=1e-10)
    isempty(sigma) && return 0
    σ1 = first(sigma)
    return count(σ -> σ >= epsilon * σ1, sigma)
end

function stats_of(times)
    ordered = sort(times)
    return (
        min=first(ordered),
        median=ordered[cld(length(ordered), 2)],
        max=last(ordered),
    )
end

function time_calls(f, repeats)
    bench = @benchmarkable $f()
    trial = run(bench; samples=repeats, evals=1, seconds=1.0e9)
    return stats_of(trial.times ./ 1e9)
end

function julia_gesvd_lwork(A::Matrix{ComplexF64})
    m, n = size(A)
    minmn = min(m, n)
    Ac = copy(A)
    S = Vector{Float64}(undef, minmn)
    U = Matrix{ComplexF64}(undef, m, minmn)
    VT = Matrix{ComplexF64}(undef, minmn, n)
    work = ComplexF64[0]
    rwork = Vector{Float64}(undef, 5minmn)
    info = Ref{LinearAlgebra.BlasInt}()
    ccall(
        (:zgesvd_64_, LinearAlgebra.BLAS.libblastrampoline), Cvoid,
        (Ref{UInt8}, Ref{UInt8}, Ref{LinearAlgebra.BlasInt}, Ref{LinearAlgebra.BlasInt},
         Ptr{ComplexF64}, Ref{LinearAlgebra.BlasInt}, Ptr{Float64}, Ptr{ComplexF64},
         Ref{LinearAlgebra.BlasInt}, Ptr{ComplexF64}, Ref{LinearAlgebra.BlasInt}, Ptr{ComplexF64},
         Ref{LinearAlgebra.BlasInt}, Ptr{Float64}, Ref{LinearAlgebra.BlasInt}, Clong, Clong),
        'S', 'S', m, n, Ac, max(1, stride(Ac, 2)), S, U, max(1, stride(U, 2)),
        VT, max(1, stride(VT, 2)), work, LinearAlgebra.BlasInt(-1), rwork, info, 1, 1,
    )
    info[] == 0 || error("Julia gesvd workspace query failed with info=$(info[])")
    return Int(real(work[1]))
end

function replay_julia(label, matrix, repeats)
    epsilon = 1e-10
    rows = NamedTuple[]
    lwork = julia_gesvd_lwork(matrix)
    for (name, alg) in (("julia_gesvd", LinearAlgebra.QRIteration()), ("julia_gesdd", LinearAlgebra.DivideAndConquer()))
        F = LinearAlgebra.svd(matrix; alg)
        rel = norm(matrix - F.U * Diagonal(F.S) * F.Vt) / max(norm(matrix), eps())
        rank = ace_rank(F.S, epsilon)
        timed = time_calls(repeats) do
            LinearAlgebra.svd(matrix; alg)
        end
        push!(rows, (;
            label, implementation=name, m=size(matrix, 1), n=size(matrix, 2),
            lda=stride(matrix, 2), lwork=name == "julia_gesvd" ? lwork : "", abi="ILP64",
            median_s=timed.median, min_s=timed.min, max_s=timed.max,
            rel_recon=rel, retained_rank=rank, samples=repeats,
            sigma=collect(F.S),
        ))
    end
    i = ITensors.Index(size(matrix, 1), "L")
    j = ITensors.Index(size(matrix, 2), "R")
    tensor = ITensors.ITensor(matrix, i, j)
    U, S, V = ITensors.svd(tensor, i; cutoff=0.0, alg="qr_iteration")
    sigma = [real(S[n, n]) for n in 1:ITensors.dim(ITensors.commonind(U, S))]
    timed = time_calls(repeats) do
        ITensors.svd(tensor, i; cutoff=0.0, alg="qr_iteration")
    end
    push!(rows, (;
        label, implementation="itensors_gesvd", m=size(matrix, 1), n=size(matrix, 2),
        lda=size(matrix, 1), lwork="", abi="ILP64",
        median_s=timed.median, min_s=timed.min, max_s=timed.max,
        rel_recon=NaN, retained_rank=ace_rank(sigma, epsilon), samples=repeats,
        sigma,
    ))
    return rows
end

function parse_cpp(text)
    values = Dict{String,String}()
    for line in split(strip(text), '\n')
        parts = split(line; limit=2)
        length(parts) == 2 || continue
        values[parts[1]] = parts[2]
    end
    return values
end

function main()
    rows = load_trace()
    labels = pick_calls(rows)
    open(joinpath(PROFILE_RESULTS, "selected_calls.csv"), "w") do io
        println(io, "label,call_index")
        for (call, label) in sort(collect(labels); by=last)
            println(io, "$label,$call")
        end
    end
    println("selected $(join(["$label=$call" for (call, label) in labels], ", "))")
    ids = join(sort(collect(keys(labels))), ",")
    dump_env = copy(ENV)
    dump_env["PROBE_ONLY"] = "gesvd"
    dump_env["PROBE_DUMP_IDS"] = ids
    dump_env["PROBE_SKIP_REFERENCE"] = "1"
    dump_env["PROBE_SKIP_SELFCHECK"] = "1"
    dump_env["PROBE_RECORD_STAGES"] = "0"
    dump_env["SKIP_INSTANTIATE"] = "1"
    julia = get(ENV, "JULIA_BIN", "julia")
    project = abspath(joinpath(PROFILE_DIR, "..", ".."))
    missing = [call for call in keys(labels) if !isfile(joinpath(PROFILE_FIXTURES, "svd_$(call).bin"))]
    if isempty(missing)
        println("fixtures already present; skipping the dump build")
    else
        run(setenv(`$julia +1.12.7 -t 1 --project=$project $(joinpath(PROFILE_DIR, "run_phase1a.jl"))`, dump_env))
    end

    replay_rows = []
    sigma_ref = Dict{String,Vector{Float64}}()
    for (call, label) in labels
        path = joinpath(PROFILE_FIXTURES, "svd_$(call).bin")
        isfile(path) || error("missing fixture $path")
        matrix = read_matrix(path)
        repeats = size(matrix, 1) * size(matrix, 2) < 256 ? 21 : 7
        julia_rows = replay_julia(label, matrix, repeats)
        append!(replay_rows, julia_rows)
        sigma_ref[label] = julia_rows[1].sigma
        exe = joinpath(PROFILE_DIR, "cpp", "gesvd_replay")
        out = read(`$exe $path $repeats $label`, String)
        parsed = parse_cpp(out)
        cpp_sigma = parse.(Float64, filter!(!isempty, readlines(path * ".lapacke_sigma.txt")))
        rel_sv = norm(cpp_sigma[1:min(end, length(sigma_ref[label]))] .- sigma_ref[label][1:min(end, length(cpp_sigma))]) /
                 max(norm(sigma_ref[label]), eps())
        open(joinpath(PROFILE_LOGS(), "replay_$(label)_cpp.txt"), "w") do io
            println(io, out)
            println(io, "sigma_rel_vs_julia_gesvd $rel_sv")
        end
        push!(replay_rows, (;
            label, implementation="lapacke_gesvd",
            m=parse(Int, parsed["m"]), n=parse(Int, parsed["n"]),
            lda=parse(Int, parsed["lda"]), lwork=parsed["lwork"], abi=parsed["abi"],
            median_s=parse(Float64, parsed["median_s"]),
            min_s=parse(Float64, parsed["min_s"]),
            max_s=parse(Float64, parsed["max_s"]),
            rel_recon=parse(Float64, parsed["recon"]),
            retained_rank=count(σ -> σ >= 1e-10 * cpp_sigma[1], cpp_sigma),
            samples=parse(Int, parsed["samples"]),
            sigma=cpp_sigma,
            sigma_rel_vs_julia=rel_sv,
        ))
    end
    open(joinpath(PROFILE_RESULTS, "replay.csv"), "w") do io
        println(io, "label,implementation,m,n,lda,lwork,abi,median_s,min_s,max_s,rel_recon,retained_rank,samples,sigma_rel_vs_julia_gesvd")
        for row in replay_rows
            rel = get(row, :sigma_rel_vs_julia, "")
            println(io, join((
                row.label, row.implementation, row.m, row.n, row.lda, row.lwork, row.abi,
                row.median_s, row.min_s, row.max_s, row.rel_recon, row.retained_rank, row.samples, rel,
            ), ","))
        end
    end
    println("wrote $(joinpath(PROFILE_RESULTS, "replay.csv"))")
    return nothing
end

PROFILE_LOGS() = joinpath(PROFILE_DIR, "logs")

main()
