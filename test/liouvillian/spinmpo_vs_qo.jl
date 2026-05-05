using Test
using LinearAlgebra
using ProcessTensors
using ITensors
using QuantumOptics

const ATOL = 1e-10
const RTOL = 1e-10

function pt_hamiltonian_opsum(N::Int; J::Float64=0.7, Δ::Float64=0.0, h::Float64=0.3)
    os_H = OpSum()
    for i in 1:N
        os_H += h, "Sx", i
    end
    for i in 1:(N - 1)
        os_H += Δ, "Sz", i, "Sz", i + 1
        os_H += J, "Sx", i, "Sx", i + 1
        os_H += J, "Sy", i, "Sy", i + 1
    end
    return os_H
end

function qo_hamiltonian(N::Int; J::Float64=0.7, Δ::Float64=0.0, h::Float64=0.3)
    b = SpinBasis(1 // 2)
    bt = tensor(fill(b, N)...)

    sx = sigmax(b) / 2
    sy = sigmay(b) / 2
    sz = sigmaz(b) / 2

    H = 0.0 * identityoperator(bt)
    if N == 1
        H += h * sx
    else
        for i in 1:N
            H += h * embed(bt, i, sx)
        end
        for i in 1:(N - 1)
            H += Δ * embed(bt, [i, i + 1], (sz ⊗ sz))
            H += J * embed(bt, [i, i + 1], (sx ⊗ sx))
            H += J * embed(bt, [i, i + 1], (sy ⊗ sy))
        end
    end
    return H, b, bt
end

function jump_site_rates(N::Int, scenario::Symbol; gamma_base::Float64=0.31, gamma_scale::Float64=0.03)
    """
    Return (site, gamma_up, gamma_down) tuples for each decay site.
    gamma_up: decay rate for S+ (raising) operator
    gamma_down: decay rate for S- (lowering) operator
    Ensure: gamma_up + gamma_down = constant total dissipation
    """
    if scenario === :edge
        if N == 1
            return [(1, gamma_base, 1.0 - gamma_base)]
        end
        return [(1, gamma_base, 1.0 - gamma_base), (N, gamma_base + 0.16, 1.0 - (gamma_base + 0.16))]
    elseif scenario === :all
        return [(i, gamma_base + gamma_scale * i, 1.0 - (gamma_base + gamma_scale * i)) for i in 1:N]
    else
        throw(ArgumentError("Unknown scenario: $scenario"))
    end
end

function pt_jump_ops(site_rates)
    """
    Convert (site, gamma_up, gamma_down) to ProcessTensors format.
    Returns: [(rate, operator_name, site), ...]
    S+ (raising): decay rate = γ_up
    S- (lowering): decay rate = γ_down
    """
    return [(γ, op, i) for (i, γ_up, γ_down) in site_rates for (γ, op) in [(γ_up, "S+"), (γ_down, "S-")]]
end

function qo_jump_ops(bt, b, site_rates)
    """
    Convert (site, gamma_up, gamma_down) to QuantumOptics format.
    S+ (raising): σ+ with rate sqrt(γ_up)
    S- (lowering): σ- with rate sqrt(γ_down)
    """
    if typeof(bt) <: SpinBasis
        @assert length(site_rates) == 1
        i, γ_up, γ_down = only(site_rates)
        return [sqrt(γ_up) * sigmap(b), sqrt(γ_down) * sigmam(b)]
    else
        return [sqrt(γ) * embed(bt, i, op) for (i, γ_up, γ_down) in site_rates for (γ, op) in [(γ_up, sigmap(b)), (γ_down, sigmam(b))]]
    end
end

function spin_hilbert_matrix_to_mpo(M::AbstractMatrix{<:Number}, physical_sites)
    N = length(physical_sites)
    dims = vcat(dim.(prime.(physical_sites)), dim.(physical_sites))
    T = ITensor(reshape(ComplexF64.(M), Tuple(dims)), prime.(physical_sites)..., physical_sites...)
    return ProcessTensors.MPO(T, physical_sites)
end

function basis_operator_matrix(d::Int, a::Int, b::Int)
    E = zeros(ComplexF64, d, d)
    E[a, b] = 1.0
    return E
end

function build_operator_bases(physical_sites, liouv_sites_shared, bt_qo)
    d = prod(dim.(physical_sites))
    d2 = d * d

    basis_pt = Vector{Any}(undef, d2)
    basis_qo = Vector{Any}(undef, d2)

    # Build the exact same operator basis E_ab in both libraries.
    for b in 1:d
        for a in 1:d
            q = a + (b - 1) * d
            E = basis_operator_matrix(d, a, b)
            basis_pt[q] = to_liouville(spin_hilbert_matrix_to_mpo(E, physical_sites); sites=liouv_sites_shared)
            basis_qo[q] = Operator(bt_qo, bt_qo, E)
        end
    end

    return basis_pt, basis_qo
end

function run_case(N::Int, scenario::Symbol; J::Float64=0.7, Δ::Float64=0.0, h::Float64=0.3, gamma_base::Float64=0.31, gamma_scale::Float64=0.03)
    # One place where all knobs are collected for this case.
    physical_sites = siteinds("S=1/2", N)
    liouv_sites_shared = liouv_sites(physical_sites)

    # Build ProcessTensors Liouvillian MPO from Hamiltonian + jumps.
    os_H = pt_hamiltonian_opsum(N; J=J, Δ=Δ, h=h)
    site_rates = jump_site_rates(N, scenario; gamma_base=gamma_base, gamma_scale=gamma_scale)
    jumps_pt = pt_jump_ops(site_rates)
    L_pt_mpo = MPO_Liouville(os_H, liouv_sites_shared; jump_ops=jumps_pt)

    # Build the QuantumOptics Liouvillian from the same physical model.
    H_qo, b_qo, bt_qo = qo_hamiltonian(N; J=J, Δ=Δ, h=h)
    jumps_qo = qo_jump_ops(bt_qo, b_qo, site_rates)
    L_qo = liouvillian(H_qo, jumps_qo)

    # Use operator basis elements as in/out states: <E_out|L|E_in>.
    basis_pt, basis_qo = build_operator_bases(physical_sites, liouv_sites_shared, bt_qo)
    d = prod(dim.(physical_sites))
    d2 = d * d

    max_element_err = 0.0
    worst_in = 1
    worst_out = 1
    max_coeff_scale = 0.0

    # For each input basis operator, compare all output amplitudes.
    for q_in in 1:d2
        ρ_pt_in = basis_pt[q_in]
        ρ_qo_in = basis_qo[q_in]

        σ_pt = apply(L_pt_mpo, ρ_pt_in)
        σ_qo = L_qo * ρ_qo_in

        # Sanity check: both outputs should be traceless (since L is a Liouvillian)
        @test abs(tr(to_hilbert(σ_pt))) ≤ 1e-9
        @test abs(tr(σ_qo)) ≤ 1e-9

        # This loop checks one full column of the superoperator matrix.
        for q_out in 1:d2
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

    # Final pass/fail uses the worst coefficient mismatch seen in this case.
    tol = ATOL + RTOL * max(1.0, max_coeff_scale)
    @test max_element_err ≤ tol

    if max_element_err > tol
        @info "Operator-basis element mismatch" N scenario max_element_err worst_in worst_out tol
    end
end

# Define parameter sweep: three dissipation rate regimes (low, med, high)
const DISSIPATION_CONFIGS = [
    (0.2, "low"),      # Weak dissipation
    (0.31, "med"),     # Medium dissipation
    (0.45, "high"),    # Strong dissipation
]

# Define integrability: Δ=0 is integrable (free fermions), Δ≠0 breaks integrability
const INTEGRABILITY_CONFIGS = [
    (0.0, "off"),  # Δ=0: integrable (no Sz_i Sz_i+1 coupling)
    (0.5, "on"),   # Δ=0.5: non-integrable
]

@testset "liouvillian.jl: spin chain with edge dissipation" begin
    for N in 1:4
        for (Δ_val, Δ_label) in INTEGRABILITY_CONFIGS
            for (gamma_base, gamma_label) in DISSIPATION_CONFIGS
                test_label = "N=$N, Sz_iSz_i+1 $Δ_label, dissipation=$gamma_label"
                @testset "$test_label" begin
                    run_case(N, :edge; Δ=Δ_val, gamma_base=gamma_base)
                end
            end
        end
    end
end

@testset "liouvillian.jl: spin chain with bulk dissipation" begin
    for N in 1:4
        for (Δ_val, Δ_label) in INTEGRABILITY_CONFIGS
            for (gamma_base, gamma_label) in DISSIPATION_CONFIGS
                test_label = "N=$N, Sz_iSz_i+1 $Δ_label, dissipation=$gamma_label"
                @testset "$test_label" begin
                    run_case(N, :all; Δ=Δ_val, gamma_base=gamma_base)
                end
            end
        end
    end
end
