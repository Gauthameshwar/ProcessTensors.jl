# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: test/networks/test_algebra.jl
# Contributor: Gauthameshwar S.
#
# Tests that compact network algebra helpers forward to ITensorMPS and rewrap
# results.
#
# Run with:
#   julia --project=. test/runtests.jl

using Test
using ITensors
import ITensorMPS
using ITensors.Ops: Prod, Trotter
using ProcessTensors

@testset "API surface: compact network export boundary" begin
    for name in (
        :apply, :contract, :add,
        :random_mps, :random_mpo, :outer, :projector,
        :inner, :dot, :norm, :expect, :correlation_matrix, :entropy, :tr,
        :add!,
    )
        @test name ∈ names(ProcessTensors)
    end

    @test ProcessTensors.apply === ITensorMPS.apply
    @test ProcessTensors.inner === ITensorMPS.inner
    @test ProcessTensors.add! === ITensorMPS.add!

    s = siteinds("S=1/2", 4)
    ls = liouv_sites(s)
    psi_h = random_mps(s; linkdims=2)
    rho_l = to_liouville(to_dm(psi_h); sites=ls)
    L = liouvillian_mpo(OpSum() + (0.1, "Sz", 1, "Sz", 2), ls)
    @test apply(L, rho_l) isa MPS{Liouville}
    @test ProcessTensors.space(apply(L, rho_l)) === Liouville
    @test ProcessTensors.space(psi_h) === Hilbert

    core = ITensorMPS.orthogonalize(psi_h.core, 2)
    @test core isa ITensorMPS.MPS
    @test MPS{Hilbert}(core) isa MPS{Hilbert}
end

@testset "algebra forwarding API" begin
    @testset "MPS algebra methods (spin and boson)" begin
        spin_sites = siteinds("S=1/2", 4)
        boson_sites = siteinds("Boson", 3; dim=3)

        m_spin = MPS(spin_sites, fill("Up", length(spin_sites)))
        m_spin_2 = MPS(spin_sites, fill("Dn", length(spin_sites)))
        op_spin = projector(m_spin)

        m_boson = MPS(boson_sites, fill("0", length(boson_sites)))
        m_boson_2 = MPS(boson_sites, fill("1", length(boson_sites)))
        op_boson = projector(m_boson)

        os_spin = OpSum()
        os_spin += 0.1, "Sz", 1, "Sz", 2
        dt = 0.05
        lazy_gates = exp(-im * dt * os_spin; alg=Trotter{2}())
        gate_prod = Prod{ITensor}(lazy_gates, collect(spin_sites))
        gates = collect(ITensor, only(gate_prod.args))
        @test_nowarn apply(gates[1], m_spin)
        @test_nowarn apply(gates, m_spin)
        @test_nowarn apply(gate_prod, m_spin)

        @test_nowarn apply(op_spin, m_spin)
        @test_nowarn contract(op_spin, m_spin)
        @test_nowarn add(m_spin, m_spin_2)
        @test_nowarn add(m_spin.core, m_spin)
        @test_nowarn m_spin + m_spin_2
        @test_nowarn m_spin - m_spin_2
        @test_nowarn 2.0 * m_spin
        @test_nowarn m_spin * 2.0

        os_boson = OpSum()
        os_boson += 0.1, "N", 1, "N", 2
        lazy_gates_b = exp(-im * dt * os_boson; alg=Trotter{2}())
        gate_prod_b = Prod{ITensor}(lazy_gates_b, collect(boson_sites))
        gates_b = collect(ITensor, only(gate_prod_b.args))
        @test_nowarn apply(gates_b[1], m_boson)
        @test_nowarn apply(gates_b, m_boson)
        @test_nowarn apply(gate_prod_b, m_boson)

        @test_nowarn apply(op_boson, m_boson)
        @test_nowarn contract(op_boson, m_boson)
        @test_nowarn add(m_boson, m_boson_2)
        @test_nowarn add(m_boson.core, m_boson)
        @test_nowarn m_boson + m_boson_2
        @test_nowarn m_boson - m_boson_2
        @test_nowarn 2.0 * m_boson
        @test_nowarn m_boson * 2.0
    end
end
