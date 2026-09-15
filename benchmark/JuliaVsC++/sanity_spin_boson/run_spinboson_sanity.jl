# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# Compare Julia zipup_cpp and C++ ACE dynamics for a 2-mode Lorentzian
# independent-boson model against exact exp(-im*t*H) evolution.
# Spectral parameters follow Table S.3.1; N_E=2 keeps the Hilbert space
# at dim 2 * 5^2 = 50.
#
# Run with:
#   bash benchmark/JuliaVsC++/run_sanity_cpp_julia_ed.sh

include(joinpath(@__DIR__, "..", "common.jl"))

const HERE = @__DIR__
const RESULTS = joinpath(HERE, "results")
const PARAM = joinpath(HERE, "spinboson_sanity.param")
const ACE_BIN = get(ENV, "ACE_BIN", joinpath(dirname(dirname(dirname(@__DIR__))), "ACE", "bin", "ACE"))

const N_MODES = 2
const LOCAL_DIM = 5
const C = 0.1
const KBT = 0.5
const DT = 0.1
const FINAL_TIME = 2.0
const NSTEPS = round(Int, FINAL_TIME / DT)
const CUTOFF = 1e-7
const COMPRESSION = Symbol(get(ENV, "JULIA_VS_CPP_COMPRESSION", "zipup_cpp"))

function one_site_density_matrix(ρ)
    tensor = foldl(*, ρ)
    site = only(filter(i -> plev(i) == 0 && hastags(i, "Site"), inds(tensor)))
    return ComplexF64.(Array(tensor, prime(site), site))
end

function ace_to_itensor_density(ρ)
    return ρ[[2, 1], [2, 1]]
end

function parse_ace_density(path)
    times = Float64[]
    states = Matrix{ComplexF64}[]
    for raw in eachline(path)
        fields = split(strip(raw))
        length(fields) >= 9 || continue
        values = parse.(Float64, fields[2:9])
        push!(times, parse(Float64, fields[1]))
        push!(
            states,
            ace_to_itensor_density(
                [
                    ComplexF64(values[1], values[2]) ComplexF64(values[3], values[4])
                    ComplexF64(values[5], values[6]) ComplexF64(values[7], values[8])
                ],
            ),
        )
    end
    isempty(states) && error("C++ ACE produced no density matrices in $path.")
    return times, states
end

function build_julia_trajectory()
    system_sites = siteinds("S=1/2", 1)
    H = OpSum()
    H += 1.0, "Sx", 1
    system = with_logger(NullLogger()) do
        spin_system(system_sites, H)
    end
    bath = lorentzian_spinboson_bath(KBT; n_modes=N_MODES, local_dim=LOCAL_DIM, C)
    ρsystem = to_dm(MPS(system_sites, ["Dn"]))
    pt = with_logger(NullLogger()) do
        build_process_tensor(
            system;
            method=ACE(cutoff=CUTOFF, compression=COMPRESSION),
            environment=bath,
            dt=DT,
            nsteps=NSTEPS,
            alg=Exact(),
            sys_alg=Trotter{2}(),
            combine_alg=Trotter{2}(),
            progress=false,
            verbose=false,
        )
    end
    trajectory = evolve(pt, ρsystem; progress=false)
    times = collect(0:DT:FINAL_TIME)
    states = vcat(
        [one_site_density_matrix(ρsystem)],
        [one_site_density_matrix(ρ) for ρ in trajectory.states_hilbert],
    )
    length(states) == length(times) || error("Julia trajectory/time length mismatch.")
    return times, states, maxlinkdim(pt.core)
end

function build_cpp_trajectory()
    isfile(ACE_BIN) || error("C++ ACE binary not found at $ACE_BIN.")
    output = joinpath(RESULTS, "cpp_rho.dat")
    log = joinpath(RESULTS, "cpp_ace.log")
    open(log, "w") do io
        run(pipeline(`$ACE_BIN $PARAM -outfile $output`, stdout=io, stderr=io))
    end
    return parse_ace_density(output)
end

creation(M) = begin
    a = zeros(ComplexF64, M, M)
    for n in 1:(M - 1)
        a[n + 1, n] = sqrt(n)
    end
    return a
end

function embed(op, site, dims)
    factors = [site == k ? op : Matrix{ComplexF64}(I, dims[k], dims[k]) for k in eachindex(dims)]
    return reduce(kron, factors)
end

function exact_model()
    frequencies, couplings = lorentzian_mode_grid(N_MODES; C)
    dims = [2; fill(LOCAL_DIM, N_MODES)]
    σx = ComplexF64[0 1; 1 0]
    proj_up = ComplexF64[1 0; 0 0]
    H = 0.5 * embed(σx, 1, dims)
    for k in 1:N_MODES
        ωk = frequencies[k]
        gk = couplings[k]
        a = creation(LOCAL_DIM)
        n = Diagonal(ComplexF64.(0:(LOCAL_DIM - 1)))
        H .+= ωk .* embed(n, k + 1, dims)
        x = a + a'
        H .+= gk .* (embed(proj_up, 1, dims) * embed(x, k + 1, dims))
        if abs(ωk) > 1e-12
            H .+= (gk^2 / ωk) .* embed(proj_up, 1, dims)
        end
    end

    down = ComplexF64[0, 1]
    ρ0 = down * down'
    for (ωk, _) in zip(frequencies, couplings)
        occupations = 0:(LOCAL_DIM - 1)
        weights = exp.(-ωk .* occupations ./ KBT)
        weights ./= sum(weights)
        ρk = Diagonal(ComplexF64.(weights))
        ρ0 = kron(ρ0, ρk)
    end
    return Hermitian(H), ρ0, dims
end

function partial_trace_bath(ρfull, dims)
    denv = prod(dims[2:end])
    ρsystem = zeros(ComplexF64, 2, 2)
    for s1 in 1:2, s2 in 1:2, env in 1:denv
        row = (s1 - 1) * denv + env
        col = (s2 - 1) * denv + env
        ρsystem[s1, s2] += ρfull[row, col]
    end
    return ρsystem
end

function exact_trajectory(times)
    H, ρ0, dims = exact_model()
    decomposition = eigen(H)
    return map(times) do t
        U = decomposition.vectors *
            Diagonal(exp.(-1im .* t .* decomposition.values)) *
            decomposition.vectors'
        partial_trace_bath(U * ρ0 * U', dims)
    end
end

function write_density_data(path, times, states; label)
    open(path, "w") do io
        println(io, "# $label")
        println(io, "# t re00 im00 re01 im01 re10 im10 re11 im11")
        for (t, ρ) in zip(times, states)
            @printf(
                io,
                "%.16e %.16e %.16e %.16e %.16e %.16e %.16e %.16e %.16e\n",
                t,
                real(ρ[1, 1]),
                imag(ρ[1, 1]),
                real(ρ[1, 2]),
                imag(ρ[1, 2]),
                real(ρ[2, 1]),
                imag(ρ[2, 1]),
                real(ρ[2, 2]),
                imag(ρ[2, 2]),
            )
        end
    end
end

relative_frobenius(ρ, reference) = norm(ρ - reference) / norm(reference)

function assert_matching_times(reference_times, candidate_times, name)
    length(reference_times) == length(candidate_times) ||
        error("$name has $(length(candidate_times)) times; expected $(length(reference_times)).")
    maximum(abs.(reference_times .- candidate_times)) < 1e-12 ||
        error("$name time grid does not match the ED grid.")
end

function write_comparison_data(path, times, julia_errors, cpp_errors, direct_errors)
    open(path, "w") do io
        println(io, "# Relative Frobenius error ||rho-rho_ED||_F / ||rho_ED||_F")
        println(io, "# t julia_vs_ed cpp_vs_ed julia_vs_cpp")
        for values in zip(times, julia_errors, cpp_errors, direct_errors)
            @printf(io, "%.16e %.16e %.16e %.16e\n", values...)
        end
    end
end

function main()
    COMPRESSION in (:zipup_cpp, :canonzip) ||
        error("Unsupported Julia compression $COMPRESSION.")
    mkpath(RESULTS)
    pin_compute_threads!()

    println("2-mode Lorentzian spin-boson sanity check")
    println("  N_modes=$N_MODES  M=$LOCAL_DIM  C=$C  kBT=$KBT")
    println("  dt=$DT  te=$FINAL_TIME  cutoff=$CUTOFF")
    println("  Julia compression=$COMPRESSION")
    println("  ED uses direct 50×50 exp(-im*t*H), with no time stepping")

    println("Building and evolving Julia PT")
    times, julia_states, julia_maxdim = build_julia_trajectory()
    println("  Julia max PT bond dimension = $julia_maxdim")

    println("Building and evolving C++ PT")
    cpp_times, cpp_states = build_cpp_trajectory()
    assert_matching_times(times, cpp_times, "C++")

    println("Computing exact full-system trajectory")
    ed_states = exact_trajectory(times)

    julia_errors = [
        relative_frobenius(ρ, exact) for (ρ, exact) in zip(julia_states, ed_states)
    ]
    cpp_errors = [
        relative_frobenius(ρ, exact) for (ρ, exact) in zip(cpp_states, ed_states)
    ]
    direct_errors = [
        relative_frobenius(ρj, ρc) for (ρj, ρc) in zip(julia_states, cpp_states)
    ]

    write_density_data(
        joinpath(RESULTS, "ed_rho.dat"),
        times,
        ed_states;
        label="Exact 2-mode independent-boson reduced density matrix",
    )
    write_density_data(
        joinpath(RESULTS, "julia_rho.dat"),
        times,
        julia_states;
        label="ProcessTensors.jl reduced density matrix",
    )
    write_comparison_data(
        joinpath(RESULTS, "relative_errors.dat"),
        times,
        julia_errors,
        cpp_errors,
        direct_errors,
    )

    println("\nRelative Frobenius errors")
    println("       t       Julia/ED         C++/ED       Julia/C++")
    for values in zip(times, julia_errors, cpp_errors, direct_errors)
        @printf("  %6.2f   %12.6e   %12.6e   %12.6e\n", values...)
    end

    max_julia = maximum(julia_errors)
    max_cpp = maximum(cpp_errors)
    max_direct = maximum(direct_errors)
    @printf("\nmax Julia vs ED = %.6e\n", max_julia)
    @printf("max C++   vs ED = %.6e\n", max_cpp)
    @printf("max Julia vs C++ = %.6e\n", max_direct)

    open(joinpath(RESULTS, "summary.txt"), "w") do io
        println(io, "model=2-mode Lorentzian independent boson")
        println(io, "compression=$COMPRESSION")
        println(io, "N_modes=$N_MODES")
        println(io, "local_dim=$LOCAL_DIM")
        println(io, "C=$C")
        println(io, "kBT=$KBT")
        println(io, "dt=$DT")
        println(io, "final_time=$FINAL_TIME")
        println(io, "cutoff=$CUTOFF")
        println(io, "julia_maxdim=$julia_maxdim")
        println(io, "max_relative_frobenius_julia_vs_ed=$max_julia")
        println(io, "max_relative_frobenius_cpp_vs_ed=$max_cpp")
        println(io, "max_relative_frobenius_julia_vs_cpp=$max_direct")
    end
end

abspath(PROGRAM_FILE) == (@__FILE__) && main()
