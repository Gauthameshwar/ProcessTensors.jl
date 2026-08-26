# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: test/systems/test_testers.jl
# Contributor: Gauthameshwar S.
#
# Tests tester ancillas, tester actions, validation, and intervention schedules.
#
# Run with:
# julia --project=. test/runtests.jl

using ProcessTensors
using ProcessTensors.Instruments: resolve_tester_action
using ITensors
using Test

const TesterIns = ProcessTensors.Instruments

@testset "API surface: tester names" begin
    for name in (
        :Tester,
        :tester,
        :AbstractTesterAction,
        :TesterIdentity,
        :TesterPropagation,
        :TesterUnitary,
        :JointPropagation,
        :JointUnitary,
        :tester_identity,
        :tester_propagation,
        :tester_unitary,
        :joint_propagation,
        :joint_unitary,
        :TesterSeq,
    )
        @test name ∈ names(ProcessTensors)
    end
    @test :resolve_tester_action ∉ names(ProcessTensors)
    @test :resolve_tester_action ∈ names(ProcessTensors.Instruments)
    @test Tester <: AbstractInstrument
    @test TesterIdentity <: AbstractTesterAction
    @test nameof(typeof(tester_identity())) == :TesterIdentity
end

@testset "testers.jl: tester construction and state compatibility" begin
    sites = siteinds("Qubit", 1)
    rho0 = to_dm(MPS(sites, ["0"]))
    memory = tester(sites, rho0)

    @test memory isa Tester
    @test length(memory.sites) == 1
    @test has_tag_token(only(memory.sites), "Liouv")
    @test memory.rho0 === rho0
    @test !hasfield(typeof(memory), :H)

    sites_liouv = liouv_sites(sites)
    rho0_liouv = to_liouville(rho0; sites=sites_liouv)
    memory_liouv = tester(sites_liouv, rho0_liouv)
    @test memory_liouv.sites == sites_liouv
    @test memory_liouv.rho0 === rho0_liouv

    @test_throws ArgumentError tester(Index[], rho0)
    @test_throws ArgumentError tester(siteinds("Qubit", 2), rho0)

    boson_sites = siteinds("Boson", 1; dim=3)
    boson_rho0 = to_dm(MPS(boson_sites, ["0"]))
    @test_throws ArgumentError tester(sites, boson_rho0)

    unrelated_liouv = liouv_sites(sites)
    @test unrelated_liouv != sites_liouv
    @test_throws ArgumentError tester(unrelated_liouv, rho0_liouv)
end

@testset "testers.jl: action constructors and validation" begin
    system_sites = siteinds("Qubit", 1)
    tester_sites = siteinds("Qubit", 1)
    rho0 = to_dm(MPS(tester_sites, ["0"]))
    memory = tester(tester_sites, rho0)

    H_tester = OpSum() + (0.2, "X", 1)
    H_joint = OpSum() + (0.3, "Z", 1, "Z", 2)
    U_tester = op("H", only(tester_sites))
    U_joint = op("CNOT", only(system_sites), only(tester_sites))

    identity_action = tester_identity()
    propagation = tester_propagation(H_tester, tester_sites)
    propagation_liouv = tester_propagation(H_tester, memory.sites)
    unitary = tester_unitary(U_tester, tester_sites)
    joint_prop = joint_propagation(H_joint, system_sites, tester_sites)
    joint_gate = joint_unitary(U_joint, system_sites, tester_sites)

    @test identity_action isa TesterIdentity
    @test propagation isa TesterPropagation
    @test propagation.H === H_tester
    @test propagation_liouv.sites == memory.sites
    @test unitary isa TesterUnitary
    @test unitary.U === U_tester
    @test joint_prop isa JointPropagation
    @test joint_prop.H === H_joint
    @test joint_gate isa JointUnitary
    @test joint_gate.U === U_joint

    @test TesterIns.validate_tester_action(identity_action, memory) === nothing
    @test TesterIns.validate_tester_action(propagation, memory) === nothing
    @test TesterIns.validate_tester_action(unitary, memory) === nothing
    @test TesterIns.validate_tester_action(
        joint_prop,
        memory;
        system_sites=system_sites,
    ) === nothing
    @test TesterIns.validate_tester_action(
        joint_gate,
        memory;
        system_sites=system_sites,
    ) === nothing

    @test_throws ArgumentError tester_propagation(H_tester, Index[])
    @test_throws ArgumentError tester_unitary(U_tester, memory.sites)
    @test_throws ArgumentError tester_unitary(
        op("H", only(system_sites)),
        tester_sites,
    )
    @test_throws ArgumentError joint_propagation(
        H_joint,
        system_sites,
        system_sites,
    )
    @test_throws ArgumentError joint_unitary(
        U_joint,
        liouv_sites(system_sites),
        tester_sites,
    )

    wrong_tester_sites = siteinds("Boson", 1; dim=3)
    wrong_tester_action = tester_propagation(
        OpSum() + (0.1, "N", 1),
        wrong_tester_sites,
    )
    @test_throws ArgumentError TesterIns.validate_tester_action(
        wrong_tester_action,
        memory,
    )
    @test_throws ArgumentError TesterIns.validate_tester_action(
        joint_prop,
        memory;
        system_sites=wrong_tester_sites,
    )
    @test_throws ArgumentError TesterIns.validate_tester_action(
        propagation,
        memory;
        system_sites=system_sites,
    )
end

@testset "testers.jl: schedules resolve and replace actions" begin
    sites = siteinds("Qubit", 1)
    Hx = OpSum() + (0.2, "X", 1)
    Hz = OpSum() + (0.3, "Z", 1)
    default = tester_identity()
    action_x = tester_propagation(Hx, sites)
    action_z = tester_propagation(Hz, sites)
    action_u = tester_unitary(op("H", only(sites)), sites)

    seq = TesterSeq(default, 4)
    @test seq.default === default
    @test resolve_tester_action(seq, 0) === default
    @test resolve_tester_action(seq, 3) === default

    @test add!(seq, action_x, 1) === seq
    @test add!(seq, action_u, 2) === seq
    @test resolve_tester_action(seq, 1) === action_x
    @test resolve_tester_action(seq, 2) === action_u

    @test add!(seq, action_z, 1) === seq
    @test resolve_tester_action(seq, 1) === action_z
    @test length(seq.entries) == 2

    keyword_seq = TesterSeq(nsteps=2, entries=Dict(0 => action_x))
    @test resolve_tester_action(keyword_seq, 0) === action_x
    @test resolve_tester_action(keyword_seq, 1) isa TesterIdentity

    @test_throws ArgumentError add!(seq, action_x, -1)
    @test_throws ArgumentError add!(seq, action_x, 5)
    @test_throws ArgumentError resolve_tester_action(seq, -1)
    @test_throws ArgumentError TesterSeq(default, -1)
    @test_throws ArgumentError TesterSeq(default, 2; entries=Dict(3 => action_x))

    instrument_seq = InstrumentSeq(default=identity_operation(), nsteps=2)
    memory = tester(sites, to_dm(MPS(sites, ["0"])))
    @test_throws ArgumentError add!(instrument_seq, memory, 1)
    @test_throws ArgumentError add!(instrument_seq, action_x, 1)
end

@testset "testers.jl: show output and object isolation" begin
    tester_sites = siteinds("Qubit", 1)
    rho0 = to_dm(MPS(tester_sites, ["0"]))
    memory = tester(tester_sites, rho0)
    seq = TesterSeq(nsteps=2)

    memory_output = sprint(show, memory)
    identity_output = sprint(show, tester_identity())
    seq_output = sprint(show, seq)
    @test occursin("ProcessTensors.Tester", memory_output)
    @test occursin("sites: 1", memory_output)
    @test occursin("tester_identity", identity_output)
    @test occursin("TesterSeq", seq_output)
    @test occursin("nsteps=2", seq_output)

    system_sites = siteinds("Qubit", 1)
    system = qubit_system(system_sites)
    pt = build_process_tensor(system; dt=0.05, nsteps=2, progress=false)

    memory_sites_before = copy(memory.sites)
    rho0_before = memory.rho0
    system_sites_before = copy(system.sites)
    system_H_before = system.H
    pt_system_before = pt.system
    pt_core_before = pt.core

    action = tester_propagation(OpSum() + (0.1, "X", 1), tester_sites)
    add!(seq, action, 1)

    @test memory.sites == memory_sites_before
    @test memory.rho0 === rho0_before
    @test system.sites == system_sites_before
    @test system.H === system_H_before
    @test pt.system === pt_system_before
    @test pt.core === pt_core_before
end
