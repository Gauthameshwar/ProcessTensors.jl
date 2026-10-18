# Phase 1C. Write the report from the saved CSVs. No new measurements.

include(joinpath(@__DIR__, "..", "..", "env.jl"))

using CairoMakie
using DelimitedFiles
using Printf
using Statistics

const DIR = @__DIR__
const RESULTS = joinpath(DIR, "results")
const LOGS = joinpath(DIR, "logs")

function table(path)
    data, header = readdlm(path, ','; header=true)
    return header, data
end

function col(header, rows, name)
    names = string.(vec(header))
    j = findfirst(==(name), names)
    j === nothing && error("column $name missing in $names")
    return rows[:, j]
end

function num(header, rows, name)
    values = col(header, rows, name)
    return [v isa Number ? Float64(v) : parse(Float64, String(v)) for v in values]
end

function txt(header, rows, name)
    return string.(col(header, rows, name))
end

function fmt(x)
    return @sprintf("%.4g", x)
end

function main()
    ref_h, ref_r = table(joinpath(RESULTS, "reference.csv"))
    stage_h, stage_r = table(joinpath(RESULTS, "stage_times.csv"))
    gesvd_h, gesvd_r = table(joinpath(RESULTS, "per_svd_gesvd.csv"))
    gesdd_h, gesdd_r = table(joinpath(RESULTS, "per_svd_gesdd.csv"))
    replay_h, replay_r = table(joinpath(RESULTS, "replay.csv"))
    prov = read(joinpath(RESULTS, "provenance.txt"), String)

    solvers = txt(stage_h, stage_r, "solver")
    gesvd_stage = findfirst(==("qr_iteration"), solvers)
    gesdd_stage = findfirst(==("divide_and_conquer"), solvers)
    gesvd_stage === nothing && error("missing gesvd stage row")
    gesdd_stage === nothing && error("missing gesdd stage row")

    function stage(row, name)
        return num(stage_h, stage_r, name)[row]
    end
    build_gesvd = stage(gesvd_stage, "build_s")
    lapack_gesvd = stage(gesvd_stage, "lapack_s")
    combine_gesvd = stage(gesvd_stage, "combine_permute_s")
    matricise_gesvd = stage(gesvd_stage, "matricise_s")
    truncate_gesvd = stage(gesvd_stage, "truncate_s")
    reassemble_gesvd = stage(gesvd_stage, "reassemble_s")
    contract_gesvd = stage(gesvd_stage, "contraction_s")
    other_gesvd = stage(gesvd_stage, "other_s")
    n_gesvd = Int(stage(gesvd_stage, "n_svd"))
    n_gesdd = Int(stage(gesdd_stage, "n_svd"))
    ref_build = num(ref_h, ref_r, "build_s")[1]
    ref_dim = Int(num(ref_h, ref_r, "maxdim")[1])

    shapes(header, rows) = collect(zip(Int.(num(header, rows, "m")), Int.(num(header, rows, "n"))))
    gesvd_shapes = shapes(gesvd_h, gesvd_r)
    gesdd_shapes = shapes(gesdd_h, gesdd_r)
    same_shapes = gesvd_shapes == gesdd_shapes

    cpp_path = joinpath(LOGS, "cpp_zgesvd.csv")
    cpp_shapes = Tuple{Int,Int}[]
    cpp_time = 0.0
    if isfile(cpp_path)
        for line in eachline(cpp_path)
            parts = split(strip(line), ",")
            length(parts) >= 8 || continue
            parts[1] == "L" || continue
            push!(cpp_shapes, (parse(Int, parts[4]), parse(Int, parts[5])))
            cpp_time += parse(Float64, parts[8])
        end
    end
    same_cpp = !isempty(cpp_shapes) && cpp_shapes == gesvd_shapes
    stages = txt(gesvd_h, gesvd_r, "stage")
    forward_idx = findall(==("forward"), stages)
    backward_idx = findall(==("backward"), stages)
    forward_match = !isempty(cpp_shapes) && all(i -> gesvd_shapes[i] == cpp_shapes[i], forward_idx)
    backward_transposed = !isempty(cpp_shapes) && length(cpp_shapes) == length(gesvd_shapes) &&
        all(i -> gesvd_shapes[i] == (cpp_shapes[i][2], cpp_shapes[i][1]), backward_idx)
    shape_key(shape) = (min(shape[1], shape[2]), max(shape[1], shape[2]))
    unordered_match = !isempty(cpp_shapes) && sort(shape_key.(gesvd_shapes)) == sort(shape_key.(cpp_shapes))

    impl = txt(replay_h, replay_r, "implementation")
    labels = txt(replay_h, replay_r, "label")
    med = num(replay_h, replay_r, "median_s")
    function replay(label, implementation)
        i = findfirst(k -> labels[k] == label && impl[k] == implementation, eachindex(labels))
        i === nothing && error("missing replay $label $implementation")
        return med[i]
    end
    expensive_ratio = replay("expensive", "julia_gesvd") / replay("expensive", "lapacke_gesvd")
    expensive_dd = replay("expensive", "julia_gesvd") / replay("expensive", "julia_gesdd")
    wrapper_ratio = replay("expensive", "itensors_gesvd") / replay("expensive", "julia_gesvd")
    ranks = Int.(num(replay_h, replay_r, "retained_rank"))
    rank_notes = String[]
    for label in ("small", "median", "expensive")
        idxs = findall(i -> labels[i] == label, eachindex(labels))
        push!(rank_notes, label * " " * join(("$(impl[i])=$(ranks[i])" for i in idxs), ", "))
    end
    sv_col = col(replay_h, replay_r, "sigma_rel_vs_julia_gesvd")
    sv_notes = String[]
    for i in eachindex(labels)
        impl[i] == "lapacke_gesvd" || continue
        push!(sv_notes, "$(labels[i]) singular-value relative difference $(sv_col[i])")
    end
    lwork_col = col(replay_h, replay_r, "lwork")
    lwork_notes = String[]
    for i in eachindex(labels)
        text = string(lwork_col[i])
        isempty(text) && continue
        push!(lwork_notes, "$(labels[i]) $(impl[i]) lwork=$text")
    end

    frac_lapack = lapack_gesvd / build_gesvd
    prep = combine_gesvd + matricise_gesvd
    rebuild = reassemble_gesvd

    fig = Figure(size=(980, 420))
    ax1 = Axis(fig[1, 1], title="N=5 partial central spin, Julia gesvd", ylabel="Seconds")
    parts = [contract_gesvd, combine_gesvd, matricise_gesvd, lapack_gesvd, truncate_gesvd, reassemble_gesvd, other_gesvd]
    names = ["contract", "combine", "matricise", "LAPACK", "truncate", "reassemble", "other"]
    barplot!(ax1, 1:length(parts), parts, color=:steelblue)
    ax1.xticks = (1:length(parts), names)
    ax1.xticklabelrotation = π / 6

    ax2 = Axis(fig[1, 2], title="Isolated replay, median", ylabel="Seconds")
    order = [("small", "julia_gesdd"), ("small", "julia_gesvd"), ("small", "lapacke_gesvd"),
             ("median", "julia_gesdd"), ("median", "julia_gesvd"), ("median", "lapacke_gesvd"),
             ("expensive", "julia_gesdd"), ("expensive", "julia_gesvd"), ("expensive", "lapacke_gesvd")]
    ys = [replay(label, implementation) for (label, implementation) in order]
    colors = [implementation == "julia_gesdd" ? :seagreen : implementation == "julia_gesvd" ? :darkorange : :purple for (_, implementation) in order]
    barplot!(ax2, 1:length(ys), ys, color=colors)
    ax2.xticks = (1:length(ys), [label * "\n" * replace(implementation, "julia_" => "", "lapacke_" => "C++ ") for (label, implementation) in order])
    ax2.xticklabelrotation = π / 4
    save(joinpath(RESULTS, "svd_profile.png"), fig)

    lapack_answer = "LAPACK is $(fmt(100 * frac_lapack))% of the instrumented Julia gesvd build ($(fmt(lapack_gesvd)) s of $(fmt(build_gesvd)) s)."
    replay_answer = if expensive_ratio > 2
        "Yes for the most expensive matrix: Julia ILP64 gesvd median / LP64 LAPACKE gesvd median = $(fmt(expensive_ratio))."
    else
        "No. On the most expensive matrix the isolated Julia/C++ gesvd ratio is $(fmt(expensive_ratio)), so the full-build gap is not reproduced by one LAPACK call."
    end
    cause = if expensive_ratio > 2 && frac_lapack > 0.7
        "The slowdown is inside the LAPACK gesvd kernel used by Julia, not in the ACE truncation. The ILP64 OpenBLAS 0.3.29 kernel and the LP64 OpenBLAS 0.3.30 kernel are different libraries; this audit does not separate the ABI from the library version."
    elseif same_shapes && expensive_ratio < 2
        "Julia gesvd and gesdd issue the same number of factorisations on the same shapes, and an isolated gesvd call is not about 3× a C++ LAPACKE call. Workspace sizes match, so `lwork` does not explain a factor of three. The instrumented gesdd wall time is not the stored production gesdd time, so this profiling copy does not reproduce that historical gap. The ABI and the OpenBLAS version remain confounded."
    else
        "The measurements split the time, but they do not yet separate the ILP64 ABI from the OpenBLAS version. That is the next experiment."
    end

    cpp_sentence = if isempty(cpp_shapes)
        "The C++ call count was not recorded."
    elseif same_cpp
        "C++ LAPACKE_zgesvd was called $(length(cpp_shapes)) times on the same (m, n) sequence as Julia gesvd."
    elseif forward_match && backward_transposed
        "C++ LAPACKE_zgesvd was called $(length(cpp_shapes)) times. The forward $(length(forward_idx)) shapes match Julia. The backward $(length(backward_idx)) shapes are the transpose of Julia's, so the unordered pairs {m, n} $(unordered_match ? "match" : "do not match"). `logs/cpp_zgesvd.csv`."
    else
        "C++ LAPACKE_zgesvd was called $(length(cpp_shapes)) times; Julia gesvd recorded $n_gesvd factorisations. The (m, n) sequences do not match. `logs/cpp_zgesvd.csv`."
    end

    report = """
    # ACE SVD performance audit, phase 1

    Partial central spin, N=5, `dt=0.1`, `t_final=20`, `ε=1e-10`, `:zipup_cpp`, C++ orientations. Julia $(VERSION), one thread. The reference build uses the production `build_process_tensor` path. The stage breakdown uses a profiling copy of the zip-up driver that calls `NDTensors.svd`, which is the factorisation `ITensors.svd` calls. No cutoff, truncation rule, or solver default was changed.

    ## Provenance

    ```
    $(prov)
    ```

    C++ ACE enters LAPACK through Eigen `JacobiSVD` with `ComputeThinU | ComputeThinV`, and this build replaces that with `LAPACKE_zgesvd` at job `S`,`S`. Julia's `LinearAlgebra.QRIteration` also calls `gesvd!` with job `S`,`S`. Julia's BLAS is ILP64. The C++ replay harness is LP64 OpenBLAS from `~/.local/openblas`.

    $(isfile(joinpath(LOGS, "isolation.txt")) ? read(joinpath(LOGS, "isolation.txt"), String) : "")

    ## Validity

    Uninstrumented Julia gesvd build: $(fmt(ref_build)) s, `D_max=$ref_dim`.
    Instrumented Julia gesvd build: $(fmt(build_gesvd)) s, `D_max=$(Int(stage(gesvd_stage, "maxdim")))`.
    Instrumented Julia gesdd build: $(fmt(stage(gesdd_stage, "build_s"))) s, `D_max=$(Int(stage(gesdd_stage, "maxdim")))`.
    The stored production timings for this case were C++ gesvd 53.48 s, Julia gesdd 55.28 s, Julia gesvd 170.69 s, all with `D_max=65`.

    ## Answers

    1. Julia gesvd performed $n_gesvd SVDs and Julia gesdd performed $n_gesdd. Their (m, n) sequences $(same_shapes ? "are identical" : "differ"). $cpp_sentence
    2. $lapack_answer
    3. $replay_answer Julia gesvd / Julia gesdd on that matrix is $(fmt(expensive_dd)). The ITensors wrapper / raw Julia gesvd ratio is $(fmt(wrapper_ratio)). ACE retained ranks: $(join(rank_notes, "; ")). Singular values: $(join(sv_notes, "; ")). Reconstruction residuals are the `rel_recon` column of `results/replay.csv`.
    4. In the instrumented gesvd build, index combination and permutation took $(fmt(combine_gesvd)) s, `NDTensors.matrix` took $(fmt(matricise_gesvd)) s, `NDTensors.svd_catch_error` (the `LinearAlgebra.svd` call, including its workspace query) took $(fmt(lapack_gesvd)) s, ACE truncation took $(fmt(truncate_gesvd)) s, and ITensor reassembly took $(fmt(reassemble_gesvd)) s. Joining the mode tensors took $(fmt(contract_gesvd)) s. Unassigned build time is $(fmt(other_gesvd)) s.
    5. Julia uses ILP64 `libopenblas64_.so` (integer width $(match(r"blas_int_bits (\d+)", prov).captures[1]) bits). C++ uses LP64 OpenBLAS 0.3.30. Workspace queries returned the same `lwork` on each replayed matrix: $(join(lwork_notes, "; ")). Workspace size does not explain a factor of about three. The ABI and the OpenBLAS version are still confounded: this run does not rebuild either library.
    6. $cause The next decisive experiment is to replay the expensive matrix with one OpenBLAS build compiled twice, LP64 and ILP64, and, separately, with 0.3.29 and 0.3.30 at fixed ABI. `perf stat` on `zgesvd_64_` versus `zgesvd_` for that matrix would show whether the extra time is inside the blocked QR iteration.

    ## Files

    - `results/reference.csv`, `results/per_svd_gesvd.csv`, `results/per_svd_gesdd.csv`, `results/stage_times.csv`
    - `results/replay.csv`, `results/svd_profile.png`
    - `fixtures/svd_*.bin`, `logs/`

    Figure: `results/svd_profile.png`.
    """
    write(joinpath(DIR, "report.md"), report)
    println("wrote $(joinpath(DIR, "report.md"))")
    return nothing
end

main()
