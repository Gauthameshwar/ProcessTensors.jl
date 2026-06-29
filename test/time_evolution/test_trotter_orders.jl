using ProcessTensors
using ITensors
using ITensors.Ops: Trotter
using LinearAlgebra
using Test

@testset "ProcessTensors: Yoshida Trotter API loaded" begin
    @test isdefined(ProcessTensors, :trotter_order)
    @test ProcessTensors.trotter_order(Trotter{4}()) == 4
end

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

function _exact_unitary_apply(ψ0, os_H::OpSum, sites, dt::Real)
    H_mpo = MPO(os_H, sites)
    H_dense = foldl(*, H_mpo)
    U = exp(-im * dt * H_dense)
    return apply(U, ψ0)
end

@testset "trotter_gates: higher-order support" begin
    sites = siteinds("S=1/2", 2)
    H = _test_hamiltonian(2)
    dt = 0.05
    ψ0 = MPS(sites, n -> isodd(n) ? "Up" : "Dn")
    ψ_exact = _exact_unitary_apply(ψ0, H, sites, dt)

    errs = Dict{Int, Float64}()
    gate_counts = Dict{Int, Int}()

    for order in (1, 2, 4, 6)
        alg = Trotter{order}()
        gates = trotter_gates(H, sites, -im * dt; alg=alg)
        ψ_trotter = apply(gates, copy(ψ0); maxdim=128, cutoff=1e-12)
        gate_counts[order] = length(gates)
        errs[order] = norm(ψ_trotter - ψ_exact) / norm(ψ_exact)
    end

    @test gate_counts[2] == 2 * gate_counts[1]
    @test gate_counts[4] == 3 * gate_counts[2]
    @test gate_counts[6] == 3 * gate_counts[4]

    @test errs[2] < errs[1]
    @test errs[4] < errs[2]
    @test errs[6] < errs[4]
    @test errs[4] < 1e-10
end

@testset "trotter_gates: unsupported odd orders" begin
    sites = siteinds("S=1/2", 2)
    H = _test_hamiltonian(2)
    @test_throws ArgumentError trotter_gates(H, sites, -im * 0.05; alg=Trotter{3}())
end

@testset "trotter_gates: nsteps repetition matches ITensors scaling" begin
    sites = siteinds("S=1/2", 2)
    H = _test_hamiltonian(2)
    dt = 0.05

    gates_one = trotter_gates(H, sites, -im * dt; alg=Trotter{2}())
    gates_two = trotter_gates(H, sites, -im * dt; alg=Trotter{2}(2))

    @test length(gates_two) == 2 * length(gates_one)
end

function _loglog_slope(x, y)
    x̄ = sum(x) / length(x)
    ȳ = sum(y) / length(y)
    return sum((x .- x̄) .* (y .- ȳ)) / sum((x .- x̄) .^ 2)
end

@testset "Trotter{4} one-step error scales as O(dt^5)" begin
    sites = siteinds("S=1/2", 2)
    H = _test_hamiltonian(2)
    ψ0 = MPS(sites, n -> isodd(n) ? "Up" : "Dn")

    dts = [0.08, 0.04, 0.02, 0.01]
    errs = Float64[]

    for dt in dts
        ψ_exact = _exact_unitary_apply(ψ0, H, sites, dt)
        gates = trotter_gates(H, sites, -im * dt; alg=Trotter{4}())
        ψ_trotter = apply(gates, copy(ψ0); maxdim=128, cutoff=1e-12)
        push!(errs, norm(ψ_trotter - ψ_exact) / norm(ψ_exact))
    end

    p = _loglog_slope(log.(dts), log.(errs))

    @test 4.6 ≤ p ≤ 5.4
end

@testset "tebd: higher-order Hilbert evolution" begin
    sites = siteinds("S=1/2", 2)
    H = _test_hamiltonian(2)
    dt = 0.05
    T = 0.05
    ψ0 = MPS(sites, n -> isodd(n) ? "Up" : "Dn")
    ψ_exact = _exact_unitary_apply(ψ0, H, sites, T)

    ψ2 = tebd(ψ0, H, dt, T; alg=Trotter{2}(), maxdim=128, cutoff=1e-12)
    ψ4 = tebd(ψ0, H, dt, T; alg=Trotter{4}(), maxdim=128, cutoff=1e-12)

    err2 = norm(ψ2 - ψ_exact) / norm(ψ_exact)
    err4 = norm(ψ4 - ψ_exact) / norm(ψ_exact)

    @test err4 < err2
end
