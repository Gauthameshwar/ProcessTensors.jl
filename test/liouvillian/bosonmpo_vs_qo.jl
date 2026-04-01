using Test
using Random
using LinearAlgebra
using ProcessTensors
using ITensors
using QuantumOptics

const ATOL_BOSON = 1e-10
const RTOL_BOSON = 1e-10

const DISSIPATION_LEVELS = [
    (0.2, "low"),
    (0.4, "med"),
    (0.7, "high"),
]

const INTERACTION_LEVELS = [
    (0.0, "off"),
    (0.6, "on"),
]

function pt_boson_hamiltonian_opsum(N::Int; t_hop::Float64=1.0, g::Float64=0.6, ω::Float64=0.2)
    os_H = OpSum()
    for i in 1:N
        os_H += ω, "N", i
        os_H += g, "N", i, "N", i
        os_H += -g, "N", i
    end
    for i in 1:(N - 1)
        os_H += -t_hop, "Adag", i, "A", i + 1
        os_H += -t_hop, "A", i, "Adag", i + 1
    end
    return os_H
end

function qo_boson_hamiltonian(N::Int, D::Int; t_hop::Float64=1.0, g::Float64=0.6, ω::Float64=0.2)
    b = FockBasis(D - 1)
    bt = tensor(fill(b, N)...)

    a = destroy(b)
    adag = create(b)
    n = number(b)
    local_h = ω * n + g * (n * n - n)

    H = 0.0 * identityoperator(bt)
    if N == 1
        H += local_h
    else
        for i in 1:N
            H += embed(bt, i, local_h)
        end
        for i in 1:(N - 1)
            H += -t_hop * embed(bt, [i, i + 1], adag ⊗ a)
            H += -t_hop * embed(bt, [i, i + 1], a ⊗ adag)
        end
    end

    return H, b, bt
end

function boson_jump_site_rates(N::Int, scenario::Symbol; gamma_base::Float64=0.4, gamma_scale::Float64=0.04)
    if scenario === :edge
        if N == 1
            return [(1, gamma_base)]
        end
        return [(1, gamma_base), (N, gamma_base + 0.1)]
    elseif scenario === :all
        return [(i, gamma_base + gamma_scale * (i - 1)) for i in 1:N]
    else
        throw(ArgumentError("Unknown scenario: $scenario"))
    end
end

boson_pt_jump_ops(site_rates) = [(γ, "A", i) for (i, γ) in site_rates]

function boson_qo_jump_ops(bt, b, site_rates)
    a = destroy(b)
    if typeof(bt) <: FockBasis
        @assert length(site_rates) == 1
        return [sqrt(only(site_rates)[2]) * a]
    end
    return [sqrt(γ) * embed(bt, i, a) for (i, γ) in site_rates]
end

function boson_hilbert_matrix_to_mpo(M::AbstractMatrix{<:Number}, physical_sites)
    dims = vcat(dim.(prime.(physical_sites)), dim.(physical_sites))
    T = ITensor(reshape(ComplexF64.(M), Tuple(dims)), prime.(physical_sites)..., physical_sites...)
    return ProcessTensors.MPO(T, physical_sites)
end

function boson_basis_operator_matrix(d::Int, a::Int, b::Int)
    E = zeros(ComplexF64, d, d)
    E[a, b] = 1.0
    return E
end

function boson_build_operator_bases(physical_sites, liouv_sites_shared, bt_qo)
    d = prod(dim.(physical_sites))
    d2 = d * d

    basis_pt = Vector{Any}(undef, d2)
    basis_qo = Vector{Any}(undef, d2)

    for b in 1:d
        for a in 1:d
            q = a + (b - 1) * d
            E = boson_basis_operator_matrix(d, a, b)
            basis_pt[q] = to_liouville(boson_hilbert_matrix_to_mpo(E, physical_sites); sites=liouv_sites_shared)
            basis_qo[q] = Operator(bt_qo, bt_qo, E)
        end
    end

    return basis_pt, basis_qo
end

function sampled_indices(d2::Int; max_samples::Int=24, seed::Int=0)
    if d2 <= max_samples
        return collect(1:d2)
    end

    rng = MersenneTwister(seed)
    idx = Set([1, d2, Int(cld(d2, 2))])
    while length(idx) < max_samples
        push!(idx, rand(rng, 1:d2))
    end
    return sort!(collect(idx))
end

function run_boson_case(
    N::Int,
    D::Int,
    scenario::Symbol;
    t_hop::Float64=1.0,
    g::Float64=0.6,
    ω::Float64=0.2,
    gamma_base::Float64=0.4,
    gamma_scale::Float64=0.04,
    full_threshold::Int=256,
    max_samples::Int=24,
)
    physical_sites = siteinds("Boson", N; dim=D, conserve_qns=false)
    liouv_sites_shared = liouv_sites(physical_sites)

    os_H = pt_boson_hamiltonian_opsum(N; t_hop=t_hop, g=g, ω=ω)
    site_rates = boson_jump_site_rates(N, scenario; gamma_base=gamma_base, gamma_scale=gamma_scale)
    L_pt_mpo = MPO_Liouville(os_H, liouv_sites_shared; jump_ops=boson_pt_jump_ops(site_rates))

    H_qo, b_qo, bt_qo = qo_boson_hamiltonian(N, D; t_hop=t_hop, g=g, ω=ω)
    L_qo = liouvillian(H_qo, boson_qo_jump_ops(bt_qo, b_qo, site_rates))

    basis_pt, basis_qo = boson_build_operator_bases(physical_sites, liouv_sites_shared, bt_qo)
    d = prod(dim.(physical_sites))
    d2 = d * d

    # Full basis on small spaces; deterministic sampling on larger spaces.
    do_full = d2 <= full_threshold
    idx_in = do_full ? collect(1:d2) : sampled_indices(d2; max_samples=max_samples, seed=1000 + 10N + D)
    idx_out = do_full ? collect(1:d2) : sampled_indices(d2; max_samples=max_samples, seed=2000 + 10N + D)

    max_element_err = 0.0
    worst_in = first(idx_in)
    worst_out = first(idx_out)
    max_coeff_scale = 0.0

    for q_in in idx_in
        ρ_pt_in = basis_pt[q_in]
        ρ_qo_in = basis_qo[q_in]

        σ_pt = apply(L_pt_mpo, ρ_pt_in)
        σ_qo = L_qo * ρ_qo_in

        @test abs(tr(to_hilbert(σ_pt))) ≤ 1e-9
        @test abs(tr(σ_qo)) ≤ 1e-9

        for q_out in idx_out
            ρ_pt_out = basis_pt[q_out]
            ρ_qo_out = basis_qo[q_out]

            coeff_pt = inner(ρ_pt_out, σ_pt)
            coeff_qo = tr(dagger(ρ_qo_out) * σ_qo)
            err = abs(coeff_pt - coeff_qo)
            max_coeff_scale = max(max_coeff_scale, abs(coeff_qo))

            if err > max_element_err
                max_element_err = err
                worst_in = q_in
                worst_out = q_out
            end
        end
    end

    tol = ATOL_BOSON + RTOL_BOSON * max(1.0, max_coeff_scale)
    @test max_element_err ≤ tol

    if max_element_err > tol
        @info "Boson operator-basis mismatch" N D scenario g gamma_base max_element_err worst_in worst_out tol
    end
end

@testset "liouvillian.jl: bosonic cavity MPO vs QO parity" begin
    @testset "single-site cavity with dissipation" begin
        for D in (2, 5, 15)
            for (gamma, gamma_label) in DISSIPATION_LEVELS
                @testset "N=1, levels=$D, dissipation=$gamma_label" begin
                    run_boson_case(1, D, :all; g=0.6, gamma_base=gamma, gamma_scale=0.0, max_samples=64)
                end
            end
        end
    end

    @testset "multi-site bosonic chain with hopping=1 and g*n(n-1)" begin
        long_mode = get(ENV, "PT_LONG_TESTS", "0") == "1"
        configs = long_mode ? [(2, 4), (2, 8), (3, 4), (3, 8)] : [(2, 2), (2, 4), (3, 4)]

        for (N, D) in configs
            for scenario in (:edge, :all)
                scenario_label = scenario === :edge ? "edge dissipation" : "bulk dissipation"
                for (g_val, g_label) in INTERACTION_LEVELS
                    for (gamma, gamma_label) in DISSIPATION_LEVELS
                        @testset "N=$N, levels=$D, $scenario_label, interaction=$g_label, dissipation=$gamma_label" begin
                            gamma_scale = scenario === :all ? 0.04 : 0.0
                            max_samples = N == 2 ? 24 : 16
                            run_boson_case(
                                N,
                                D,
                                scenario;
                                g=g_val,
                                gamma_base=gamma,
                                gamma_scale=gamma_scale,
                                max_samples=max_samples,
                            )
                        end
                    end
                end
            end
        end
    end
end
