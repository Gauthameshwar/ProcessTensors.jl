using ProcessTensors
using ITensors
using LinearAlgebra
using Test

if !(@isdefined dense_liouvillian_matrix)
    include(joinpath(@__DIR__, "tebd_test_utils.jl"))
end

# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

# Function to return the mean of an Array
function mean(x::AbstractArray{<:Number})
    return sum(x) / length(x)
end

# For N=1: read propagator as dense matrix matching instruments / dense_liouvillian_matrix convention.
# `liouvillian_propagator_itensor` uses unprimed `s` = output (ket), `prime(s)` = input (bra).
# Array order is (input, output) = (prime(s), s), consistent with `instrument_itensor` tests.
function _dense_from_1site_propagator(U::ITensor, s::Index)
    d2 = dim(s)
    return reshape(Array(U, prime(s), s), d2, d2)
end

# Build E_{ab} basis state as a Liouville MPS on `phys` sites vectorized to `liouv`.
function _liouv_basis_mps(a::Int, b::Int, phys, liouv)
    d = prod(dim.(phys))
    E = zeros(ComplexF64, d, d)
    E[a, b] = 1.0
    return to_liouville(hilbert_matrix_to_mpo(E, phys); sites=liouv)
end

# Build the full d²×d² dense matrix of the propagator by applying U to each basis state.
# Works for any N; uses the same TEBD-based apply available in the module.
function _dense_from_propagator_apply(U::ITensor, phys, liouv)
    d = prod(dim.(phys))
    d2 = d * d
    U_mat = zeros(ComplexF64, d2, d2)
    for b in 1:d, a in 1:d
        q_in = a + (b - 1) * d
        ρ_in = _liouv_basis_mps(a, b, phys, liouv)
        # apply(U::ITensor, ρ::AbstractMPS) dispatches via networks/algebra.jl
        ρ_out = apply(U, ρ_in; maxdim=typemax(Int), cutoff=0.0)
        ρ_out_dense = liouville_state_to_dense(ρ_out, phys)
        # Output column q_in = vec(ρ_out) in column-major order
        for bb in 1:d, aa in 1:d
            q_out = aa + (bb - 1) * d
            U_mat[q_out, q_in] = ρ_out_dense[aa, bb]
        end
    end
    return U_mat
end

# Small TFIM-like Hamiltonian (nearest-neighbour XX + transverse field).
function _test_hamiltonian(N::Int)
    H = OpSum()
    for j in 1:N
        H += 0.5, "Sx", j
        H += 0.3, "Sz", j
    end
    for j in 1:(N - 1)
        H += 0.4, "Sx", j, "Sx", j + 1
    end
    return H
end

# -------------------------------------------------------------------------
# N = 1 tests
# -------------------------------------------------------------------------

@testset "liouvillian_propagator_itensor: N=1, closed, Exact()" begin
    phys = siteinds("S=1/2", 1)
    liouv = liouv_sites(phys)
    H = _test_hamiltonian(1)
    dt = 0.07

    L_dense = dense_liouvillian_matrix(H, [], phys, liouv)
    U_ed = exp(dt * L_dense)
    U = liouvillian_propagator_itensor(H, liouv, dt; alg=Exact())
    U_mat = _dense_from_1site_propagator(U, liouv[1])

    @test relative_frobenius_error(U_mat, U_ed) ≤ 1e-10
end

@testset "liouvillian_propagator_itensor: N=1, dissipative, Exact()" begin
    phys = siteinds("S=1/2", 1)
    liouv = liouv_sites(phys)
    H = _test_hamiltonian(1)
    jump_ops = []
    dt = 0.07

    L_dense = dense_liouvillian_matrix(H, jump_ops, phys, liouv)
    U_ed = exp(dt * L_dense)
    U = liouvillian_propagator_itensor(H, liouv, dt; alg=Exact(), jump_ops=jump_ops)
    U_mat = _dense_from_1site_propagator(U, liouv[1])

    @test relative_frobenius_error(U_mat, U_ed) ≤ 1e-10
end

@testset "liouvillian_propagator_itensor: N=1, Trotter{2}() vs Exact()" begin
    phys = siteinds("S=1/2", 1)
    liouv = liouv_sites(phys)
    H = _test_hamiltonian(1)
    jump_ops = []
    dt = 0.05

    L_dense = dense_liouvillian_matrix(H, jump_ops, phys, liouv)
    U_ed = exp(dt * L_dense)

    U_exact = liouvillian_propagator_itensor(H, liouv, dt; alg=Exact(), jump_ops=jump_ops)
    U_trotter = liouvillian_propagator_itensor(H, liouv, dt; alg=Trotter{2}(), jump_ops=jump_ops)

    U_exact_mat = _dense_from_1site_propagator(U_exact, liouv[1])
    U_trotter_mat = _dense_from_1site_propagator(U_trotter, liouv[1])

    @test relative_frobenius_error(U_exact_mat, U_ed) ≤ 1e-10
    @test relative_frobenius_error(U_trotter_mat, U_ed) ≤ 1e-4
end

# -------------------------------------------------------------------------
# N = 2 tests
# -------------------------------------------------------------------------

@testset "liouvillian_propagator_itensor: N=2, closed, Exact()" begin
    phys = siteinds("S=1/2", 2)
    liouv = liouv_sites(phys)
    H = _test_hamiltonian(2)
    dt = 0.05

    L_dense = dense_liouvillian_matrix(H, [], phys, liouv)
    U_ed = exp(dt * L_dense)
    U = liouvillian_propagator_itensor(H, liouv, dt; alg=Exact())
    U_mat = _dense_from_propagator_apply(U, phys, liouv)

    @test relative_frobenius_error(U_mat, U_ed) ≤ 1e-8
end

@testset "liouvillian_propagator_itensor: N=2, dissipative, Exact()" begin
    phys = siteinds("S=1/2", 2)
    liouv = liouv_sites(phys)
    H = _test_hamiltonian(2)
    jump_ops = [(0.15, "S-", 1)]
    dt = 0.05

    L_dense = dense_liouvillian_matrix(H, jump_ops, phys, liouv)
    U_ed = exp(dt * L_dense)
    U = liouvillian_propagator_itensor(H, liouv, dt; alg=Exact(), jump_ops=jump_ops)
    U_mat = _dense_from_propagator_apply(U, phys, liouv)

    @test relative_frobenius_error(U_mat, U_ed) ≤ 1e-8
end

@testset "liouvillian_propagator_itensor: N=2, Trotter{2}() vs Exact()" begin
    phys = siteinds("S=1/2", 2)
    liouv = liouv_sites(phys)
    H = _test_hamiltonian(2)
    dt = 0.04

    L_dense = dense_liouvillian_matrix(H, [], phys, liouv)
    U_ed = exp(dt * L_dense)
    U_exact = liouvillian_propagator_itensor(H, liouv, dt; alg=Exact())
    U_trotter = liouvillian_propagator_itensor(H, liouv, dt; alg=Trotter{2}())

    U_exact_mat = _dense_from_propagator_apply(U_exact, phys, liouv)
    U_trotter_mat = _dense_from_propagator_apply(U_trotter, phys, liouv)

    @test relative_frobenius_error(U_exact_mat, U_ed) ≤ 1e-8
    @test relative_frobenius_error(U_trotter_mat, U_ed) ≤ 1e-4
end

# -------------------------------------------------------------------------
# N = 3 tests
# -------------------------------------------------------------------------

@testset "liouvillian_propagator_itensor: N=3, closed, Exact()" begin
    phys = siteinds("S=1/2", 3)
    liouv = liouv_sites(phys)
    H = _test_hamiltonian(3)
    dt = 0.04

    L_dense = dense_liouvillian_matrix(H, [], phys, liouv)
    U_ed = exp(dt * L_dense)
    U = liouvillian_propagator_itensor(H, liouv, dt; alg=Exact())
    U_mat = _dense_from_propagator_apply(U, phys, liouv)

    @test relative_frobenius_error(U_mat, U_ed) ≤ 1e-8
end

@testset "liouvillian_propagator_itensor: N=3, Trotter{2}() vs Exact()" begin
    phys = siteinds("S=1/2", 3)
    liouv = liouv_sites(phys)
    H = _test_hamiltonian(3)
    dt = 0.03

    L_dense = dense_liouvillian_matrix(H, [], phys, liouv)
    U_ed = exp(dt * L_dense)
    U_exact = liouvillian_propagator_itensor(H, liouv, dt; alg=Exact())
    U_trotter = liouvillian_propagator_itensor(H, liouv, dt; alg=Trotter{2}())

    U_exact_mat = _dense_from_propagator_apply(U_exact, phys, liouv)
    U_trotter_mat = _dense_from_propagator_apply(U_trotter, phys, liouv)

    @test relative_frobenius_error(U_exact_mat, U_ed) ≤ 1e-8
    @test relative_frobenius_error(U_trotter_mat, U_ed) ≤ 1e-4
end

@testset "Trotter{2} one-step error scales as O(dt^3)" begin
    phys = siteinds("S=1/2", 2)
    liouv = liouv_sites(phys)
    H = _test_hamiltonian(2)

    dts = [0.08, 0.04, 0.02, 0.01]
    errs = Float64[]

    for dt in dts
        L_dense = dense_liouvillian_matrix(H, [], phys, liouv)
        U_ed = exp(dt * L_dense)

        U_trotter = liouvillian_propagator_itensor(
            H,
            liouv,
            dt;
            alg=Trotter{2}(),
        )

        U_trotter_mat = _dense_from_propagator_apply(U_trotter, phys, liouv)
        push!(errs, relative_frobenius_error(U_trotter_mat, U_ed))
    end

    # Fit log(err) = p log(dt) + c.
    x = log.(dts)
    y = log.(errs)
    p = sum((x .- mean(x)) .* (y .- mean(y))) / sum((x .- mean(x)).^2)

    # For one-step second-order Trotter, expect p ≈ 3.
    @test 2.6 ≤ p ≤ 3.4
end