using ProcessTensors
using ITensors
using LinearAlgebra
using Test

# Single-site analytic check in vec(ρ) basis:
# vec(ρ) = [ρ00, ρ10, ρ01, ρ11]^T
# For H = (ω/2) Sz and jump L = S- with rate γ:
# d/dt vec(ρ) = [ -γρ00,
#                 ( iω/2 - γ/2)ρ10,
#                 (-iω/2 - γ/2)ρ01,
#                  γρ00 ]

function expected_action_formula(v::AbstractVector{<:Number}, ω::Real, γ::Real)
    ρ00, ρ10, ρ01, ρ11 = ComplexF64.(v)
    return ComplexF64[
        -γ * ρ00,
        (1im * ω / 2 - γ / 2) * ρ10,
        (-1im * ω / 2 - γ / 2) * ρ01,
        γ * ρ00,
    ]
end

function basis_vec(k::Int)
    v = zeros(ComplexF64, 4)
    v[k] = 1.0
    return v
end

@testset "liouvillian.jl: single-spin analytical formula check" begin
    ω = 1.3
    γ = 0.4

    physical_sites = siteinds("S=1/2", 1)
    sL = liouv_sites(physical_sites)

    os_H = OpSum()
    os_H += (ω / 2), "Sz", 1
    jump_ops = [(γ, "S-", 1)]

    L_mpo = MPO_Liouville(os_H, sL; jump_ops=jump_ops)
    L = L_mpo[1]
    c = sL[1]

    @testset "basis vector $label" for (k, label) in enumerate(("[1,0,0,0]", "[0,1,0,0]", "[0,0,1,0]", "[0,0,0,1]"))
        vin = basis_vec(k)
        ρvec = ITensor(vin, c)
        vout_num = Array(L * ρvec, prime(c))
        vout_expected = expected_action_formula(vin, ω, γ)
        err = norm(vout_num - vout_expected)

        @test err < 1e-11
    end

    @testset "linearity with arbitrary coefficients" begin
        v = ComplexF64[0.3, -0.2 + 0.1im, 0.7 - 0.4im, 0.9]
        v_num = Array(L * ITensor(v, c), prime(c))
        v_expected = expected_action_formula(v, ω, γ)
        err = norm(v_num - v_expected)

        @test err < 1e-11
    end
end
