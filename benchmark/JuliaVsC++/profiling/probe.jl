# Profiling-only copy of the ACE zip-up split.
# The factorisation is NDTensors.svd, the same routine ITensors.svd calls.
# Stage timers sit around that call. They do not change truncation or the solver.

using ITensors
using LinearAlgebra

const NDTensors = ITensors.NDTensors

const PROBE_LIB = Ref("")

function kernel_reset!()
    isempty(PROBE_LIB[]) && return
    ccall((:svd_kernel_reset, PROBE_LIB[]), Cvoid, ())
    return nothing
end

kernel_calls() = isempty(PROBE_LIB[]) ? 0 : Int(ccall((:svd_kernel_calls, PROBE_LIB[]), UInt64, ()))
kernel_total_s() = isempty(PROBE_LIB[]) ? 0.0 : ccall((:svd_kernel_total_ns, PROBE_LIB[]), UInt64, ()) / 1e9
kernel_last_m() = isempty(PROBE_LIB[]) ? 0 : Int(ccall((:svd_kernel_last_m, PROBE_LIB[]), Int64, ()))
kernel_last_n() = isempty(PROBE_LIB[]) ? 0 : Int(ccall((:svd_kernel_last_n, PROBE_LIB[]), Int64, ()))
kernel_last_lda() = isempty(PROBE_LIB[]) ? 0 : Int(ccall((:svd_kernel_last_lda, PROBE_LIB[]), Int64, ()))
kernel_last_lwork() = isempty(PROBE_LIB[]) ? 0 : Int(ccall((:svd_kernel_last_lwork, PROBE_LIB[]), Int64, ()))
kernel_last_jobu() = isempty(PROBE_LIB[]) ? '?' : Char(ccall((:svd_kernel_last_jobu, PROBE_LIB[]), Cint, ()))
kernel_last_jobvt() = isempty(PROBE_LIB[]) ? '?' : Char(ccall((:svd_kernel_last_jobvt, PROBE_LIB[]), Cint, ()))
kernel_last_kind() = isempty(PROBE_LIB[]) ? '?' : Char(ccall((:svd_kernel_last_kind, PROBE_LIB[]), Cint, ()))

gc_alloc() = Int(Base.gc_num().allocd)

function probe_svd(A::ITensor, left_inds; lefttags=nothing, alg=nothing)
    left_inds = collect(ITensors.indices(left_inds))
    t0 = time_ns()
    b0 = gc_alloc()
    Lis = commoninds(A, ITensors.indices(left_inds))
    Ris = uniqueinds(A, Lis)
    Lis_original = Lis
    Ris_original = Ris
    vLα = nothing
    vRα = nothing
    if isempty(Lis_original)
        α = ITensors.trivial_index(Ris)
        vLα = onehot(ITensors.datatype(A), α => 1)
        A *= vLα
        Lis = [α]
    end
    if isempty(Ris_original)
        α = ITensors.trivial_index(Lis)
        vRα = onehot(ITensors.datatype(A), α => 1)
        A *= vRα
        Ris = [α]
    end
    CL = combiner(Lis...; dir=nothing)
    CR = combiner(Ris...; dir=nothing)
    AC = A * CR * CL
    cL = combinedind(CL)
    cR = combinedind(CR)
    if inds(AC) != (cL, cR)
        AC = permute(AC, cL, cR)
    end
    combine_s = (time_ns() - t0) / 1e9
    combine_bytes = gc_alloc() - b0

    t_mat = time_ns()
    b_mat = gc_alloc()
    matrix_AC = NDTensors.matrix(ITensors.tensor(AC))
    matricise_s = (time_ns() - t_mat) / 1e9
    matricise_bytes = gc_alloc() - b_mat
    solver = alg === nothing ? "divide_and_conquer" : String(alg)
    linear_alg = solver == "qr_iteration" ? LinearAlgebra.QRIteration() : LinearAlgebra.DivideAndConquer()
    kernel_reset!()
    t1 = time_ns()
    b1 = gc_alloc()
    USVT = NDTensors.svd_catch_error(matrix_AC; alg=linear_alg)
    lapack_s = (time_ns() - t1) / 1e9
    nd_bytes = gc_alloc() - b1 + matricise_bytes
    nd_s = matricise_s + lapack_s
    USVT === nothing && error("ACE profiling SVD failed")
    MU, MS, MV = USVT
    conj!(MV)
    P = MS .^ 2
    P, truncerr, _ = NDTensors.truncate!!(
        P; mindim=nothing, maxdim=nothing, cutoff=0.0,
        use_absolute_cutoff=nothing, use_relative_cutoff=nothing,
    )
    spec = NDTensors.Spectrum(P, truncerr)
    dS = length(P)
    if dS < length(MS)
        MU = MU[:, 1:dS]
        MS = MS[1:dS]
        MV = MV[:, 1:dS]
    end
    Ttensor = ITensors.tensor(AC)
    uind = eltype(typeof(NDTensors.inds(Ttensor)))(dS)
    vind = eltype(typeof(NDTensors.inds(Ttensor)))(dS)
    Uinds = typeof(NDTensors.inds(Ttensor))((NDTensors.ind(Ttensor, 1), uind))
    Sinds = typeof(NDTensors.inds(Ttensor))((uind, vind))
    Vinds = typeof(NDTensors.inds(Ttensor))((NDTensors.ind(Ttensor, 2), vind))
    UT = NDTensors.tensor(NDTensors.Dense(vec(MU)), Uinds)
    ST = NDTensors.tensor(NDTensors.Diag(MS), Sinds)
    VT = NDTensors.tensor(NDTensors.Dense(vec(MV)), Vinds)
    USVT = (UT, ST, VT, spec)

    utags = lefttags === nothing ? ITensors.ts"Link,u" : lefttags
    vtags = ITensors.ts"Link,v"
    t2 = time_ns()
    b2 = gc_alloc()
    UT, ST, VT, spec = USVT
    UC, S, VC = itensor(UT), itensor(ST), itensor(VT)
    u = commonind(S, UC)
    v = commonind(S, VC)
    U = UC * dag(CL)
    V = VC * dag(CR)
    U = settags(U, utags, u)
    S = settags(S, utags, u)
    S = settags(S, vtags, v)
    V = settags(V, vtags, v)
    if vLα !== nothing
        U *= dag(vLα)
    end
    if vRα !== nothing
        V *= dag(vRα)
    end
    reassemble_s = (time_ns() - t2) / 1e9
    reassemble_bytes = gc_alloc() - b2
    m_mat, n_mat = size(matrix_AC)
    meta = (;
        combine_s, combine_bytes, nd_s, nd_bytes, lapack_s, matricise_s,
        reassemble_s, reassemble_bytes,
        m=m_mat, n=n_mat, lda=max(1, m_mat),
        lwork=kernel_calls() > 0 ? kernel_last_lwork() : -1,
        jobu='S', jobvt='S',
        kind=solver == "qr_iteration" ? 'v' : 'd',
        kernel_calls=kernel_calls(),
        matrix=AC, cL, cR,
    )
    return U, S, V, spec, meta
end

function probe_split(
    source::ITensor, left_inds;
    epsilon::Real, maxdim::Integer, lefttags, svd_alg=nothing,
)
    U, S, V, spec, meta = probe_svd(source, left_inds; lefttags, alg=svd_alg)
    t0 = time_ns()
    b0 = gc_alloc()
    u = commonind(U, S)
    v = commonind(S, V)
    singular_values = [real(S[n, n]) for n in 1:dim(u)]
    nkeep = ProcessTensors._ace_relative_keep_cpp(singular_values; epsilon, maxdim)
    σ1 = first(singular_values)
    σ1 > 0 || throw(ArgumentError("ACE compression encountered a zero tensor."))
    if nkeep < dim(u)
        ut = Index(nkeep; tags=tags(u), plev=plev(u))
        vt = Index(nkeep; tags=tags(v), plev=plev(v))
        T = eltype(S)
        Ru = ITensor(T, dag(u), ut)
        Rv = ITensor(T, dag(v), vt)
        for n in 1:nkeep
            Ru[n, n] = one(T)
            Rv[n, n] = one(T)
        end
        U = U * Ru
        S = (S * Ru) * Rv
        V = V * Rv
    end
    truncate_s = (time_ns() - t0) / 1e9
    truncate_bytes = gc_alloc() - b0
    t1 = time_ns()
    b1 = gc_alloc()
    if abs(σ1 - 1) > 1e-6
        left, right = σ1 * U, (S / σ1) * V
    else
        left, right = U, S * V
    end
    ace_reassemble_s = (time_ns() - t1) / 1e9
    ace_reassemble_bytes = gc_alloc() - b1
    return left, right, merge(meta, (;
        truncate_s, truncate_bytes, ace_reassemble_s, ace_reassemble_bytes,
        retained_rank=nkeep, sigma1=σ1,
    ))
end

mutable struct ProbeLog
    io::IO
    call::Int
    dump_ids::Set{Int}
    dump_dir::String
    contraction_s::Float64
    contraction_bytes::Int
end

function write_row!(log::ProbeLog, stage, mode, timestep, contraction_s, meta, solver)
    log.call += 1
    if log.call in log.dump_ids
        matrix = Matrix{ComplexF64}(NDTensors.matrix(ITensors.tensor(meta.matrix)))
        path = joinpath(log.dump_dir, "svd_$(log.call).bin")
        open(path, "w") do io
            write(io, Int64(size(matrix, 1)))
            write(io, Int64(size(matrix, 2)))
            write(io, matrix)
        end
        open(joinpath(log.dump_dir, "svd_$(log.call).txt"), "w") do io
            println(io, "call_index $(log.call)")
            println(io, "stage $stage")
            println(io, "mode $mode")
            println(io, "timestep $timestep")
            println(io, "m $(size(matrix, 1))")
            println(io, "n $(size(matrix, 2))")
            println(io, "retained_rank $(meta.retained_rank)")
            println(io, "solver $solver")
        end
    end
    println(
        log.io,
        join((
            log.call, stage, mode, timestep, "ComplexF64",
            meta.m, meta.n, meta.lda,
            meta.m * meta.n * min(meta.m, meta.n),
            meta.retained_rank, solver, meta.jobu, meta.jobvt, meta.kind,
            meta.kernel_calls, meta.lwork,
            contraction_s, meta.combine_s, meta.matricise_s, meta.lapack_s,
            meta.truncate_s, meta.reassemble_s + meta.ace_reassemble_s,
            meta.combine_bytes + meta.nd_bytes + meta.truncate_bytes + meta.reassemble_bytes + meta.ace_reassemble_bytes,
        ), ","),
    )
    flush(log.io)
    return nothing
end

const PER_SVD_HEADER = join((
    "call_index", "stage", "mode", "timestep", "eltype", "m", "n", "lda", "cost",
    "retained_rank", "solver", "jobu", "jobvt", "kernel", "kernel_calls", "lwork",
    "contraction_s", "combine_permute_s", "matricise_s", "lapack_s", "truncate_s",
    "reassemble_s", "alloc_bytes",
), ",")

function instrumented_central_spin(system, bath; dt, nsteps, cutoff, maxdim, svd_alg, log)
    PT = ProcessTensors
    alg = ITensors.Ops.Exact()
    sys_alg = ITensors.Ops.Trotter{2}()
    combine_alg = ITensors.Ops.Trotter{2}()
    coupling_site = system.sites[1]
    cores = ITensor[]
    inputs = Index[]
    outputs = Index[]
    for k in 0:(nsteps - 1)
        in_k, out_k = PT._generate_pt_legs(coupling_site, k)
        push!(inputs, in_k)
        push!(outputs, out_k)
        push!(cores, delta(in_k, out_k))
    end
    solver = svd_alg === nothing ? "divide_and_conquer" : String(svd_alg)
    for (mode_index, mode) in enumerate(bath.modes)
        mode_cores, mode_inputs, mode_outputs = PT._ace_mode_cores(
            combine_alg, coupling_site, mode, dt, nsteps, alg,
        )
        pass_on = nothing
        for n in 1:nsteps
            t0 = time_ns()
            b0 = gc_alloc()
            PT._ace_join_core!(
                combine_alg, cores, inputs, outputs,
                mode_cores, mode_inputs, mode_outputs, n,
            )
            contraction_s = (time_ns() - t0) / 1e9
            log.contraction_s += contraction_s
            log.contraction_bytes += gc_alloc() - b0
            if pass_on !== nothing
                cores[n] *= pass_on
            end
            if n < nsteps
                right_inds = unique(vcat(
                    collect(commoninds(cores[n], cores[n + 1])),
                    collect(commoninds(cores[n], mode_cores[PT._ace_next_mode_core(combine_alg, n)])),
                ))
                left_inds = [i for i in inds(cores[n]) if i ∉ right_inds]
                cores[n], pass_on, meta = probe_split(
                    cores[n], left_inds;
                    epsilon=cutoff, maxdim, lefttags="PT,Link,tstep=$n", svd_alg,
                )
                write_row!(log, "forward", mode_index, n, contraction_s, meta, solver)
            end
        end
        for source in length(cores):-1:2
            destination = source - 1
            bond = PT._ace_fuse_bond!(cores, destination)
            source_unique = uniqueinds(cores[source], cores[destination])
            cores[source], pass_on, meta = probe_split(
                cores[source], source_unique;
                epsilon=cutoff, maxdim, lefttags=tags(bond), svd_alg,
            )
            cores[destination] *= pass_on
            write_row!(log, "backward", mode_index, source, 0.0, meta, solver)
        end
    end
    for n in 1:nsteps
        cores[n] = PT._embed_system_map(cores[n], system, inputs[n], outputs[n], dt, sys_alg)
    end
    return cores
end

function selfcheck_probe()
    i = Index(3, "i")
    j = Index(4, "j")
    k = Index(5, "k")
    A = random_itensor(ComplexF64, i, j, k)
    U1, S1, V1 = svd(A, i, j; cutoff=0.0, alg="qr_iteration")
    U2, S2, V2, _, meta = probe_svd(A, [i, j]; lefttags="Link,probe", alg="qr_iteration")
    s1 = [real(S1[n, n]) for n in 1:dim(commonind(U1, S1))]
    s2 = [real(S2[n, n]) for n in 1:dim(commonind(U2, S2))]
    length(s1) == length(s2) || error("probe SVD rank $(length(s2)) != ITensors rank $(length(s1))")
    rel = norm(s1 .- s2) / max(norm(s1), eps())
    rel < 1e-8 || error("probe SVD singular values differ from ITensors.svd by $rel")
    meta.m == 12 && meta.n == 5 || error("expected matricisation 12×5, got $(meta.m)×$(meta.n)")
    meta.kind == 'v' || error("expected gesvd kernel label, got $(meta.kind)")
    return meta
end
