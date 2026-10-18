# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# One ACE construction with LAPACK gesvd (`svd_alg="qr_iteration"`).
# The parallel driver launches one process per (model, N) point.
#
#   GESVD_MODEL=central_spin GESVD_N=5 \
#     julia +1.12.7 -t 1 --project=benchmark \
#     benchmark/JuliaVsC++/run_gesvd_case.jl
#
#   GESVD_MODEL=spinboson GESVD_N=5 \
#     julia +1.12.7 -t 1 --project=benchmark \
#     benchmark/JuliaVsC++/run_gesvd_case.jl

include(joinpath(@__DIR__, "common.jl"))

const GESVD_ALG = "qr_iteration"

function gesvd_fragment_path()
    model = get(ENV, "GESVD_MODEL", "")
    n = get(ENV, "GESVD_N", "")
    default = joinpath(RESULTS_DIR, ".gesvd_cases", "$(model)_N$(n).csv")
    return get(ENV, "GESVD_FRAGMENT", default)
end

function run_central_spin(N::Int)
    nsteps = nsteps_for(CS_FINAL_TIME, CS_DT)
    system = empty_central_spin_system()
    J = CS_J
    polarisation = "partial"
    Jk = J / N
    setup = @elapsed begin
        orientations = orientations_for(polarisation, N)
        bath = central_spin_bath(orientations; J)
    end
    Sx, Sy, Sz = total_environment_spin(orientations)
    println("=== Julia ACE gesvd  central-spin partial  N=$N  J_k=$Jk ===")
    @printf("  bath  <S> = (%.6f, %.6f, %.6f)\n", Sx, Sy, Sz)
    _, stats, χ = measure_ace_pt(
        system, bath;
        dt=CS_DT, nsteps, cutoff=CS_CUTOFF, maxdim=CS_MAXDIM,
        compression=:zipup_cpp, svd_alg=GESVD_ALG,
    )
    return (; setup, build_s=stats.t_min_s, maxdim=χ, cutoff=CS_CUTOFF)
end

function run_spinboson(N::Int)
    nsteps = nsteps_for(SB_FINAL_TIME, SB_DT)
    system = driven_tls_system()
    kBT = 0.5
    println("=== Julia ACE gesvd  spin-boson  kBT/Omega=$kBT  N_modes=$N ===")
    setup = @elapsed begin
        bath = lorentzian_spinboson_bath(kBT; n_modes=N)
    end
    _, stats, χ = measure_ace_pt(
        system, bath;
        dt=SB_DT, nsteps, cutoff=SB_CUTOFF, maxdim=SB_MAXDIM,
        compression=:zipup_cpp, svd_alg=GESVD_ALG,
    )
    return (; setup, build_s=stats.t_min_s, maxdim=χ, cutoff=SB_CUTOFF)
end

function main()
    model = get(ENV, "GESVD_MODEL", "")
    model in ("central_spin", "spinboson") || error(
        "GESVD_MODEL must be central_spin or spinboson; got $(repr(model)).",
    )
    N = parse(Int, get(ENV, "GESVD_N", ""))
    pin_compute_threads!()
    (Threads.nthreads() == 1 && BLAS.get_num_threads() == 1) || error(
        "gesvd comparison requires julia -t 1 and BLAS threads=1; got Julia=$(Threads.nthreads()) BLAS=$(BLAS.get_num_threads()).",
    )
    if get(ENV, "JULIA_VS_CPP_PRINT_PROVENANCE", "1") != "0"
        print_julia_provenance()
        print_timing_policy()
    end
    println("SVD algorithm = $GESVD_ALG (LAPACK gesvd)")
    println("CPU $(cpus_allowed())  model=$model  N=$N  BLAS=$(BLAS.get_num_threads())")

    result = model == "central_spin" ? run_central_spin(N) : run_spinboson(N)
    result.maxdim >= 1 || error("ACE build failed for $model N=$N (maxdim=$(result.maxdim)).")
    @printf(
        "  setup_s=%.6f  build_s=%.6f  maxdim=%d  svd=%s\n",
        result.setup, result.build_s, result.maxdim, GESVD_ALG,
    )

    fragment = gesvd_fragment_path()
    mkpath(dirname(fragment))
    open(fragment, "w") do io
        println(io, "model,N,julia_gesvd_build_s,julia_gesvd_maxdim,julia_gesvd_setup_s,svd_alg,cutoff")
        println(
            io,
            join(
                (model, N, result.build_s, result.maxdim, result.setup, GESVD_ALG, result.cutoff),
                ",",
            ),
        )
    end
    println("wrote $fragment")
    return nothing
end

main()
