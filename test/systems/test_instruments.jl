# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: test/systems/test_instruments.jl
# Contributor: Gauthameshwar S.
#
# Tests lazy instrument schedules, composition, and ITensor materialization.
#
# Run with:
#   julia --project=. test/runtests.jl

using ProcessTensors
using ProcessTensors.Instruments: instrument_itensor, create_instruments, resolve_instrument
using ITensors
using LinearAlgebra
using Test

if !(@isdefined hilbert_matrix_to_mpo)
    include(joinpath(@__DIR__, "..", "time_evolution", "tebd_test_utils.jl"))
end

const Ins = ProcessTensors.Instruments
const _instrument_leg_maps = Ins._instrument_leg_maps

# --- internal helper validation (Instruments submodule) ---------------------------------

@testset "Instruments.jl: low-level leg / map helpers" begin
    # Example: plev must be 0 or 1 — plev=2 is rejected.
    # Then: invalid plev throws ArgumentError.
    @test_throws ArgumentError Ins._assert_valid_leg_plev(2)
    @test Ins._assert_valid_leg_plev(0) === true
    @test Ins._assert_valid_leg_plev(1) === true

    s = siteinds("S=1/2", 1)
    L = liouv_sites(s)
    # Example: single-leg sites with two indices should fail validation.
    # Then: ArgumentError from length check.
    @test_throws ArgumentError Ins._validate_single_leg_sites("X", [L[1], L[1]], 0)

    # Example: index with tag tstep=3 — parser returns 3.
    idx = Index(dim(L[1]); tags="Liouv,tstep=3")
    @test Ins._tstep_from_site(idx) == 3
    @test Ins._tstep_from_site(L[1]) === nothing

    inp = [prime(L[1])]
    out = [L[1]]
    # Example: consistent primed/unprimed pair, no tstep tags — OK.
    @test Ins._validate_two_leg_map("Z", inp, out) === nothing
    # Example: input/output tsteps break tin == tout+1 — fail.
    bad_in = [Index(dim(L[1]); tags="Liouv,tstep=2,plev=1")]   # need prime for plev — use real prime
    bad_in2 = [prime(Index(dim(L[1]); tags="Liouv,tstep=2"))]
    bad_out2 = [Index(dim(L[1]); tags="Liouv,tstep=2")] # tout not tin-1
    @test_throws ArgumentError Ins._validate_two_leg_map("Z", bad_in2, bad_out2)
end

@testset "Instruments.jl: instrument struct validation (no dense ITensor build)" begin
    s = siteinds("S=1/2", 1)
    L = liouv_sites(s)
    psi = MPS(s, ["Up"])
    op_z = OpSum(); op_z += 1.0, "Sz", 1

    # Given: valid single-site Hilbert state and Liouville sites.
    # When: construct StatePreparation / TraceOut / ObservableMeasurement.
    # Then: constructors succeed with matching leg prime levels.
    @test state_preparation(to_liouville(to_dm(psi); sites=L)) isa StatePreparation
    @test state_preparation(to_liouville(to_dm(psi))) isa StatePreparation
    @test_throws ArgumentError state_preparation(to_liouville(to_dm(psi); sites=L), leg_plev=0)

    @test trace_out([L[1]]; leg_plev=0) isa TraceOut
    @test trace_out() isa TraceOut
    @test_throws ArgumentError trace_out([prime(L[1])]; leg_plev=1)

    @test observable_measurement(op_z, [prime(L[1])]; leg_plev=1) isa ObservableMeasurement
    @test observable_measurement(op_z, [L[1]]; leg_plev=0) isa ObservableMeasurement
    @test observable_measurement(op_z) isa ObservableMeasurement

    # Given: wrong prime level on supplied sites vs declared leg_plev.
    # Then: constructor throws ArgumentError.
    @test_throws ArgumentError state_preparation(psi, [L[1]]; leg_plev=1)

    s3 = siteinds("S=1/2", 3)
    H3 = OpSum(); H3 += 0.3, "Sz", 2
    # Given: lazy UnitaryPropagation / IdentityOperation (deferred PT legs).
    # Then: zero-length leg vectors are allowed.
    @test unitary_propagation(spin_system(s3, H3)) isa UnitaryPropagation
    @test identity_operation() isa IdentityOperation
    @test open_output() isa OpenOutput

    s1 = siteinds("S=1/2", 1)
    L1 = liouv_sites(s1)
    H1 = OpSum(); H1 += 0.5, "Sx", 1
    sys1 = spin_system(s1, H1)
    in1 = prime(L1[1])
    out0 = L1[1]
    prop_bound = unitary_propagation([in1], [out0], sys1)
    @test prop_bound.input_pt_sites == [in1]
    @test prop_bound.output_pt_sites == [out0]
    @test prop_bound.sites == sys1.sites
    id_bound = identity_operation([in1], [out0])
    @test id_bound.input_pt_sites == [in1]
    @test id_bound.output_pt_sites == [out0]
    open_bound = open_output([out0])
    @test open_bound.pt_sites == [out0]
    @test open_bound.leg_plev == 0
    @test_throws ArgumentError open_output([in1]; leg_plev=1)
end

@testset "Instruments.jl: strict constructors and removed helpers" begin
    s = siteinds("S=1/2", 1)
    L = liouv_sites(s)
    psi = MPS(s, ["Up"])
    rho = to_dm(psi)
    rhoL = to_liouville(rho; sites=L)
    op_z = OpSum(); op_z += 1.0, "Sz", 1
    sys = spin_system(s, OpSum() + (0.2, "Sz", 1))

    @test StatePreparation(rhoL, Index[], 1) isa StatePreparation
    @test TraceOut(Index[], 0) isa TraceOut
    @test OpenOutput(Index[], 0) isa OpenOutput
    @test OpenInput(Index[], 1) isa OpenInput
    @test IdentityOperation(Index[], Index[]) isa IdentityOperation
    @test OpenInOut(Index[], Index[]) isa OpenInOut
    @test UnitaryPropagation(Index[], Index[], sys.H, Index[sys.sites...]) isa UnitaryPropagation

    @test_throws ArgumentError TraceOut(Index[], 1)
    @test_throws ArgumentError OpenOutput(Index[], 1)
    @test_throws ArgumentError OpenInput(Index[], 0)
    @test_throws ArgumentError StatePreparation(rhoL, Index[], 0)

    @test_throws MethodError TraceOut()
    @test_throws MethodError IdentityOperation()
    @test_throws MethodError OpenOutput()
    @test_throws MethodError OpenInput()
    @test_throws MethodError OpenInOut()
    @test_throws MethodError UnitaryPropagation(sys)
    @test_throws MethodError StatePreparation(rho)
    @test_throws MethodError ObservableMeasurement(op_z)
end


@testset "Instruments.jl: add! generic identity" begin
    import ITensorMPS
    @test ProcessTensors.add! === ITensorMPS.add!
    @test ProcessTensors.Instruments.add! === ProcessTensors.add!
end

@testset "Instruments.jl: InstrumentSeq / resolve / add!" begin
    sys = spin_system(siteinds("S=1/2", 1), OpSum() + (0.2, "Sz", 1))
    def = unitary_propagation(sys)

    # Given: unified schedule with default only.
    # When: resolve k ≥ 1 missing from entries.
    # Then: default instrument is returned.
    seq = InstrumentSeq(def, 0)
    @test resolve_instrument(seq, 1) === def
    @test resolve_instrument(seq, 0) === nothing

    # Given: optional init absent.
    # When: resolve k == 0.
    # Then: nothing.
    @test resolve_instrument(seq, 0) === nothing

    # Given: StatePreparation at tstep 0.
    # When: resolve 0.
    # Then: that instrument is returned.
    s = siteinds("S=1/2", 1); L = liouv_sites(s)
    prep = state_preparation(to_liouville(to_dm(MPS(s, ["Dn"])); sites=L))
    add!(seq, prep, 0)
    @test resolve_instrument(seq, 0) === prep
    empty!(seq.entries)

    # Given: non–StatePreparation at tstep 0.
    # When: add!.
    # Then: ArgumentError.
    @test_throws ArgumentError add!(seq, def, 0)

    # Given: two adds at the same tstep.
    # When: add! twice.
    # Then: latest wins.
    seq2 = InstrumentSeq(identity_operation(), 0)
    add!(seq2, def, 2)
    other = unitary_propagation(sys)
    add!(seq2, other, 2)
    @test resolve_instrument(seq2, 2) === other

    # Given: second schedule; add! at step 1.
    # When: resolve 1.
    # Then: returns the instrument passed to add!.
    seq3 = InstrumentSeq(identity_operation(), 0)
    add!(seq3, def, 1)
    @test resolve_instrument(seq3, 1) === def

    # Given: seq.nsteps bound and tstep above range.
    # When: add!.
    # Then: ArgumentError.
    seq4 = InstrumentSeq(identity_operation(), 3; entries=Dict{Int,AbstractInstrument}())
    @test_throws ArgumentError add!(seq4, def, 4)

    @test_throws ArgumentError resolve_instrument(seq2, -1)

    # resolve with fallback for k ≥ 1
    seq5 = InstrumentSeq(identity_operation(), 0)
    fb = unitary_propagation(sys)
    @test resolve_instrument(seq5, 1, fb) === fb
    add!(seq5, def, 1)
    @test resolve_instrument(seq5, 1, fb) === def

    seq_kw = InstrumentSeq(; default=identity_operation(), nsteps=2, entries=Dict(1 => def))
    @test resolve_instrument(seq_kw, 1) === def
    @test occursin("InstrumentSeq", sprint(show, seq_kw))

    seq_plus = InstrumentSeq(identity_operation(), 2)
    seq_plus += (def, 2)
    @test resolve_instrument(seq_plus, 2) === def
end

@testset "Instruments.jl: _instrument_leg_maps coverage" begin
    sys = spin_system(siteinds("S=1/2", 1), OpSum() + (0.1, "Sz", 1))
    iddef = identity_operation()
    s_leg = siteinds("S=1/2", 1)
    L_leg = liouv_sites(s_leg)
    prep0 = state_preparation(to_liouville(to_dm(MPS(s_leg, ["Up"])); sites=L_leg))

    # Given: nsteps=1, default two-leg, no prep at 0.
    # When: _instrument_leg_maps.
    # Then: missing_in lists 0 (init leg), no missing_out (nsteps==1 expects no out coverage).
    seq = InstrumentSeq(iddef, 0)
    in_m, out_m, mi, mo = _instrument_leg_maps(seq, 1)
    @test mi == [0]
    @test isempty(mo)

    # Given: same with explicit StatePreparation at 0.
    # When: maps.
    # Then: missing_in empty.
    s = siteinds("S=1/2", 1); L = liouv_sites(s)
    prep = state_preparation(to_liouville(to_dm(MPS(s, ["Up"])); sites=L))
    seq_p = InstrumentSeq(iddef, 0; init=prep)
    _, _, mi2, mo2 = _instrument_leg_maps(seq_p, 1)
    @test isempty(mi2) && isempty(mo2)

    # Given: nsteps ∈ {3,4}, default Identity plus init prep at 0.
    # When: maps.
    # Then: all required in/out slots covered; terminal open legs follow API contract.
    for n in (3, 4)
        seqn = InstrumentSeq(iddef, 0; init=prep0)
        _, _, minn, mout = _instrument_leg_maps(seqn, n)
        @test isempty(minn) && isempty(mout)
    end

    # Given: sparse overrides — only step 3 set, rest default Identity, with prep at 0.
    # When: nsteps=4.
    # Then: in/out coverage complete.
    seq_s = InstrumentSeq(iddef, 0; init=prep0)
    add!(seq_s, unitary_propagation(sys), 3)
    _, _, mis, mos = _instrument_leg_maps(seq_s, 4)
    @test isempty(mis) && isempty(mos)

    # Given: last-wins at same step changes TwoLeg instrument.
    # When: leg maps.
    # Then: maps reference the replaced instrument at that step.
    seq_l = InstrumentSeq(iddef, 0)
    spA = unitary_propagation(sys); spB = unitary_propagation(sys)
    add!(seq_l, spA, 2); add!(seq_l, spB, 2)
    im, om, _, _ = _instrument_leg_maps(seq_l, 3)
    @test im[2] === spB
    @test om[1] === spB

    # Given: Observable on input leg (plev 1) at step 2 (replaces default two-leg there).
    # When: maps for nsteps=3.
    # Then: input slots stay covered via prep + other steps; output slot at tstep=1 lacks an instrument.
    op = OpSum(); op += 1.0, "Sz", 1
    seq_o = InstrumentSeq(iddef, 0; init=prep0)
    add!(seq_o, observable_measurement(op, Index[]; leg_plev=1), 2)
    _, _, mio, moo = _instrument_leg_maps(seq_o, 3)
    @test isempty(mio)
    @test moo == [1]

    # Given: no prep, Identity default, nsteps=3.
    # When: maps.
    # Then: missing_in reports 0 (documented sparse-init case).
    seq_np = InstrumentSeq(iddef, 0)
    _, _, mi_np, _ = _instrument_leg_maps(seq_np, 3)
    @test 0 in mi_np

    # Open bookkeeping coverage and pt wrapper.
    system = spin_system(s_leg, OpSum() + (0.2, "Sz", 1))
    pt = build_process_tensor(system; dt=0.05, nsteps=3)
    rho0_h = to_dm(MPS(s_leg, ["Up"]))

    seq_closed = InstrumentSeq(default=identity_operation(), nsteps=pt.nsteps)
    add!(seq_closed, state_preparation(rho0_h), 0)
    add!(seq_closed, trace_out(), pt.nsteps)
    in_a, out_a, mi_a, mo_a = _instrument_leg_maps(seq_closed, pt.nsteps)
    in_b, out_b, mi_b, mo_b = _instrument_leg_maps(pt, seq_closed)
    @test isempty(mi_a) && isempty(mo_a)
    @test mi_a == mi_b && mo_a == mo_b
    @test keys(in_a) == keys(in_b)
    @test keys(out_a) == keys(out_b)

    seq_oo = InstrumentSeq(default=identity_operation(), nsteps=pt.nsteps)
    add!(seq_oo, state_preparation(rho0_h), 0)
    add!(seq_oo, open_inout(), 1)
    add!(seq_oo, trace_out(), pt.nsteps)
    in_o, out_o, mi_o, mo_o = _instrument_leg_maps(seq_oo, pt.nsteps)
    @test isempty(mi_o) && isempty(mo_o)
    @test in_o[1] isa OpenInOut
    @test out_o[0] isa OpenInOut

    seq_out_only = InstrumentSeq(default=identity_operation(), nsteps=pt.nsteps)
    add!(seq_out_only, state_preparation(rho0_h), 0)
    add!(seq_out_only, open_output(), 1)
    # OpenOutput at step 1 claims out[0]; in[1] is missing.
    _, _, mi_only, mo_out = _instrument_leg_maps(seq_out_only, pt.nsteps)
    @test 1 in mi_only
    @test !(0 in mo_out)

    seq_in_only = InstrumentSeq(default=identity_operation(), nsteps=pt.nsteps)
    add!(seq_in_only, state_preparation(rho0_h), 0)
    add!(seq_in_only, open_input(), 1)
    _, _, mi_in, mo_in = _instrument_leg_maps(seq_in_only, pt.nsteps)
    @test 0 in mo_in
    @test !(1 in mi_in)

    seq_pair = InstrumentSeq(default=identity_operation(), nsteps=pt.nsteps)
    add!(seq_pair, state_preparation(rho0_h), 0)
    add!(seq_pair, trace_out(), 1)
    add!(seq_pair, state_preparation(to_dm(MPS(s_leg, ["Dn"]))), 2)
    add!(seq_pair, trace_out(), 3)
    _, _, mi_p, mo_p = _instrument_leg_maps(seq_pair, pt.nsteps)
    @test isempty(mi_p) && isempty(mo_p)

    @test_throws ArgumentError _instrument_leg_maps(seq_closed, 0)
end

@testset "Instruments.jl: OpenInput / OpenInOut materialization" begin
    s = siteinds("S=1/2", 1)
    L = liouv_sites(s)
    in1 = prime(L[1])
    out0 = L[1]

    @test open_input() isa OpenInput
    @test open_inout() isa OpenInOut
    @test open_input().leg_plev == 1
    @test open_output().leg_plev == 0
    @test_throws ArgumentError open_input(; leg_plev=0)
    @test_throws ArgumentError open_inout([out0], [in1])  # wrong plev order

    Tin = instrument_itensor(open_input(), [in1], 1)
    @test length(inds(Tin)) == 0
    @test isapprox(scalar(Tin), 1.0; atol=1e-12)

    Tio = instrument_itensor(open_inout(), [in1], [out0], 1)
    @test length(inds(Tio)) == 0
    @test isapprox(scalar(Tio), 1.0; atol=1e-12)

    system = spin_system(s, OpSum() + (0.3, "Sz", 1))
    pt = build_process_tensor(system; dt=0.05, nsteps=3)
    rho0_h = to_dm(MPS(s, ["Up"]))
    seq = InstrumentSeq(default=identity_operation(), nsteps=pt.nsteps)
    add!(seq, state_preparation(rho0_h), 0)
    add!(seq, open_inout(), 1)
    add!(seq, trace_out(), 3)
    instruments = create_instruments(pt, seq)
    @test length(instruments) == pt.nsteps + 1
    @test length(inds(instruments[2])) == 0

    seq_bad = InstrumentSeq(default=identity_operation(), nsteps=pt.nsteps)
    add!(seq_bad, state_preparation(rho0_h), 0)
    add!(seq_bad, open_input(), pt.nsteps)
    @test_throws ArgumentError create_instruments(pt, seq_bad)
end

@testset "Instruments.jl: instrument_itensor — StatePreparation & dense data" begin
    s_spin = siteinds("S=1/2", 1)
    Ls = liouv_sites(s_spin)
    ρh = hilbert_matrix_to_mpo(randn(ComplexF64, 2, 2), s_spin)
    ρl = to_liouville(ρh; sites=Ls)
    prep = state_preparation(ρl)
    leg_in = prime(Ls[1])
    # Given: random single-qubit density matrix as MPO{Hilbert} → Liouville MPS.
    # When: build StatePreparation ITensor at k=0 on primed Liouville leg.
    # Then: tensor matches contracted Liouville state on that leg.
    T = instrument_itensor(prep, [leg_in], 0)
    ref = contract_core(ρl.core)
    ref = replaceind(ref, Ls[1], leg_in)
    @test hassameinds(T, Index[leg_in])
    @test isapprox(vec(Array(T, leg_in)), vec(Array(ref, leg_in)))

    s_b = siteinds("Boson", 1; dim=4)
    Lb = liouv_sites(s_b)
    ρbh = hilbert_matrix_to_mpo(randn(ComplexF64, 4, 4), s_b)
    ρbl = to_liouville(ρbh; sites=Lb)
    prep_b = state_preparation(ρbl)
    leg_b = prime(Lb[1])
    Tb = instrument_itensor(prep_b, [leg_b], 0)
    refb = contract_core(ρbl.core)
    refb = replaceind(refb, Lb[1], leg_b)
    @test isapprox(vec(Array(Tb, leg_b)), vec(Array(refb, leg_b)))
end

@testset "Instruments.jl: instrument_itensor — ObservableMeasurement (spin & boson)" begin
    s = siteinds("S=1/2", 1)
    L = liouv_sites(s)
    for opname in ("Sx", "Sy", "Sz")
        os = OpSum(); os += 1.0, opname, 1
        obs = observable_measurement(os, Index[]; leg_plev=0)
        leg = L[1]
        # Given: σz / σy as OpSum observable on output leg (plev 0).
        # When: instrument_itensor at k=1.
        # Then: single Liouville index with matching dimension.
        T = instrument_itensor(obs, [leg], 1)
        @test hastags(leg, "Liouv")
        @test plev(leg) == 0
        @test hasind(T, leg)
        @test dim(leg) == 4
    end

    sb = siteinds("Boson", 1; dim=4)
    Lb = liouv_sites(sb)
    on = OpSum(); on += 1.0, "N", 1
    obs_n = observable_measurement(on, Index[]; leg_plev=0)
    Ta = instrument_itensor(obs_n, [Lb[1]], 1)
    @test hasind(Ta, Lb[1])

    oad = OpSum(); oad += 1.0, "A", 1; oad += 1.0, "Adag", 1
    obs_ad = observable_measurement(oad, Index[]; leg_plev=1)
    leg_in = prime(Lb[1])
    Tb = instrument_itensor(obs_ad, [leg_in], 2)
    @test plev(leg_in) == 1
    @test hasind(Tb, leg_in)
end

@testset "Instruments.jl: instrument_itensor — TraceOut vs vec(I)" begin
    s = siteinds("S=1/2", 1)
    L = liouv_sites(s)
    d = dim(s[1])
    I_h = Matrix{ComplexF64}(I, d, d)
    vec_id = vec(I_h)
    leg = L[1]
    # Given: spin-1/2 Liouville site (d²=4) and TraceOut on output leg.
    # When: instrument_itensor.
    # Then: dense data matches column-major vec(identity) in the Liouville basis.
    T = instrument_itensor(trace_out(), [leg], 1)
    @test hastags(leg, "Liouv")
    data = vec(Array(T, leg))
    @test length(data) == d^2
    @test isapprox(data, vec_id)
end

@testset "Instruments.jl: instrument_itensor — OpenOutput" begin
    s = siteinds("S=1/2", 1)
    L = liouv_sites(s)
    in1 = prime(L[1])
    out0 = L[1]
    open_op = open_output()
    Topen = instrument_itensor(open_op, [out0], 0)
    @test length(inds(Topen)) == 0
    @test isapprox(scalar(Topen), 1.0; atol=1e-12)

    Ttrace = instrument_itensor(trace_out([out0]), [out0], 0)
    @test length(inds(Ttrace)) == 1
    @test hasind(Ttrace, out0)

    Tid = instrument_itensor(identity_operation(), [in1], [out0], 1)
    @test length(inds(Tid)) == 2

    iddef = identity_operation()
    seq = InstrumentSeq(iddef, 2)
    add!(seq, state_preparation(to_liouville(to_dm(MPS(s, ["Up"])); sites=L)), 0)
    add!(seq, open_output(), 2)
    in_map, out_map, missing_in, missing_out = _instrument_leg_maps(seq, 2)
    @test haskey(in_map, 1)
    @test haskey(out_map, 0)
    @test isempty(missing_in)
    @test isempty(missing_out)
end

@testset "Instruments.jl: instrument_itensor — UnitaryPropagation vs exp(dt * L)" begin
    s = siteinds("S=1/2", 1)
    L = liouv_sites(s)
    H = OpSum(); H += 0.6, "Sx", 1
    sys = spin_system(s, H)
    prop = unitary_propagation(sys)
    dt = 0.04
    in1 = prime(L[1])
    out0 = L[1]
    # Given: single-site Sy-type Hamiltonian and Liouvillian MPO.
    # When: instrument_itensor (order-1 Trotter) vs dense exp(dt * L).
    # Then: maps agree at requested indices.
    Tmap = instrument_itensor(prop, [in1], [out0], 1; dt=dt, alg=Trotter{1}())
    L_mpo = liouvillian_mpo(H, L; jump_ops=[])
    T_L = contract_core(L_mpo.core)
    d = dim(L[1])
    Lmat = reshape(Array(T_L, prime(L[1]), L[1]), d, d)
    Uexp = exp(dt * Lmat)
    A = reshape(Array(Tmap, in1, out0), d, d)
    @test isapprox(A, Uexp; atol=1e-10, rtol=1e-8)

    # Given: empty Hamiltonian (constructor emits a warning by design).
    # When: UnitaryPropagation ITensor.
    # Then: delta map between in/out legs; warning is captured/asserted by @test_logs.
    sys0 = @test_warn r"SpinSystem: H is empty" spin_system(s, OpSum())
    idprop = unitary_propagation(sys0)
    Tid = instrument_itensor(idprop, [in1], [out0], 1; dt=dt, alg=Trotter{1}())
    @test isapprox(norm(Tid - delta(in1, out0)), 0.0; atol=1e-12)

    prop_bound = unitary_propagation([in1], [out0], sys)
    T_bound = instrument_itensor(prop_bound, Index[], Index[], 1; dt=dt, alg=Trotter{1}())
    @test T_bound isa ITensor
    @test hasind(T_bound, in1) && hasind(T_bound, out0)

    H_of_t(t) = OpSum() + (0.6 + 0.2t, "Sx", 1)
    prop_t = unitary_propagation(H_of_t, L)
    T_t = instrument_itensor(prop_t, [in1], [out0], 2; dt=dt, alg=Trotter{1}())
    H_mid = H_of_t(1.5dt)
    L_mid = liouvillian_mpo(H_mid, L; jump_ops=[])
    T_L_mid = contract_core(L_mid.core)
    Lmat_mid = reshape(Array(T_L_mid, prime(L[1]), L[1]), d, d)
    A_mid = reshape(Array(T_t, in1, out0), d, d)
    @test isapprox(A_mid, exp(dt * Lmat_mid); atol=1e-10, rtol=1e-8)
end

@testset "Instruments.jl: CustomTwoLegInstrument" begin
    s = siteinds("S=1/2", 1)
    L = liouv_sites(s)
    in1 = prime(L[1])
    out0 = L[1]

    @testset "ready tensor from instrument_itensor" begin
        T_ref = instrument_itensor(identity_operation(), [in1], [out0], 1)
        custom = custom_twoleg_instrument(T_ref, [in1], [out0])
        @test custom isa TwoLegInstrument
        @test occursin("CustomTwoLegInstrument", sprint(show, custom))
        T = instrument_itensor(custom, [in1], [out0], 1)
        @test isapprox(norm(T - T_ref), 0.0; atol=1e-12)

        system = spin_system(s, OpSum() + (0.1, "Sz", 1))
        pt = build_process_tensor(system; dt=0.05, nsteps=3)
        out1, in2 = coupling_times(pt, 2)
        T_step2 = instrument_itensor(custom, in2, out1, 2)
        T_ref2 = instrument_itensor(identity_operation(), in2, out1, 2)
        @test isapprox(norm(T_step2 - T_ref2), 0.0; atol=1e-12)
    end

    @testset "reindexing tensor with source legs" begin
        src_in = Index(dim(L[1]), "src_in")
        src_out = Index(dim(L[1]), "src_out")
        T_data = ITensor(1.0)
        T_data *= delta(src_in, src_out)

        custom_lazy = custom_twoleg_instrument(T_data; source_input=[src_in], source_output=[src_out])
        T_lazy = instrument_itensor(custom_lazy, [in1], [out0], 1)
        T_ref = instrument_itensor(identity_operation(), [in1], [out0], 1)
        @test isapprox(norm(T_lazy - T_ref), 0.0; atol=1e-12)

        custom_bound = custom_twoleg_instrument(
            T_data;
            source_input=[src_in],
            source_output=[src_out],
            input_pt_sites=[in1],
            output_pt_sites=[out0],
        )
        T_bound = instrument_itensor(custom_bound, [in1], [out0], 1)
        @test isapprox(norm(T_bound - T_ref), 0.0; atol=1e-12)
    end

    @testset "reindexing LeftRightOperator data" begin
        op_z = OpSum() + (1.0, "Sz", 1)
        lr = left_action(op_z, s)
        in1 = prime(L[1])
        out0 = L[1]
        T_ref = instrument_itensor(lr, [in1], [out0], 1)

        src_in = Index(dim(L[1]), "src_in")
        src_out = Index(dim(L[1]), "src_out")
        T_data = replaceinds(T_ref, in1 => src_in, out0 => src_out)
        custom = custom_twoleg_instrument(T_data; source_input=[src_in], source_output=[src_out])
        T = instrument_itensor(custom, [in1], [out0], 1)
        @test isapprox(Array(T, in1, out0), Array(T_ref, in1, out0); atol=1e-12)
    end

    @testset "validation errors" begin
        T_ref = instrument_itensor(identity_operation(), [in1], [out0], 1)
        bad_in = Index(dim(L[1]), "bad_in")
        @test_throws ArgumentError custom_twoleg_instrument(T_ref, [bad_in], [out0])
        src_in = Index(dim(L[1]), "src_in")
        src_out = Index(dim(L[1]), "src_out")
        T_data = ITensor(1.0) * delta(src_in, src_out)
        @test_throws ArgumentError custom_twoleg_instrument(
            T_data;
            source_input=[src_in],
            source_output=[src_out],
            input_pt_sites=[in1],
        )
    end

    @testset "_instrument_leg_maps and create_instruments" begin
        custom = custom_twoleg_instrument(
            instrument_itensor(identity_operation(), [in1], [out0], 1),
            [in1],
            [out0],
        )
        seq = InstrumentSeq(custom, 2)
        add!(seq, state_preparation(to_liouville(to_dm(MPS(s, ["Up"])); sites=L)), 0)
        _, _, missing_in, missing_out = _instrument_leg_maps(seq, 2)
        @test isempty(missing_in)
        @test isempty(missing_out)

        system = spin_system(s, OpSum() + (0.3, "Sz", 1))
        pt = build_process_tensor(system; dt=0.05, nsteps=2)
        rho0_h = to_dm(MPS(s, ["Up"]))
        seq_pt = InstrumentSeq(default=custom, nsteps=pt.nsteps)
        add!(seq_pt, state_preparation(rho0_h), 0)
        instruments = create_instruments(pt, seq_pt)
        @test length(instruments) == pt.nsteps + 1
        out_prev, in_curr = coupling_times(pt, 1)
        T_id = instrument_itensor(identity_operation(), in_curr, out_prev, 1)
        @test isapprox(norm(instruments[2] - T_id), 0.0; atol=1e-12)
        @test length(inds(instruments[end])) == 0
        @test isapprox(scalar(instruments[end]), 1.0; atol=1e-12)
    end

    @testset "create_instruments pairs consecutive single-leg output/input entries" begin
        system = spin_system(s, OpSum() + (0.3, "Sz", 1))
        pt = build_process_tensor(system; dt=0.05, nsteps=3)
        rho0_h = to_dm(MPS(s, ["Up"]))
        rho1_h = to_dm(MPS(s, ["Dn"]))

        seq_unpaired = InstrumentSeq(default=identity_operation(), nsteps=pt.nsteps)
        add!(seq_unpaired, state_preparation(rho0_h), 0)
        add!(seq_unpaired, trace_out(), 1)
        err = @test_throws ArgumentError create_instruments(pt, seq_unpaired)
        @test occursin("single-leg input instrument", lowercase(string(err.value)))

        seq_paired = InstrumentSeq(default=identity_operation(), nsteps=pt.nsteps)
        add!(seq_paired, state_preparation(rho0_h), 0)
        add!(seq_paired, trace_out(), 1)
        add!(seq_paired, state_preparation(rho1_h), 2)
        add!(seq_paired, trace_out(), 3)
        instruments = create_instruments(pt, seq_paired)
        @test length(instruments) == pt.nsteps + 1

        out_prev, in_curr = coupling_times(pt, 1)
        paired = trace_out() * state_preparation(rho1_h)
        expected = instrument_itensor(paired, in_curr, out_prev, 1)
        @test isapprox(norm(instruments[2] - expected), 0.0; atol=1e-12)

        out2, in2 = coupling_times(pt, 2)
        expected2 = instrument_itensor(identity_operation(), in2, out2, 2)
        @test isapprox(norm(instruments[3] - expected2), 0.0; atol=1e-12)
        @test length(inds(instruments[end])) == 1
    end
end

@testset "Instruments.jl: ProductInstrument" begin
    s = siteinds("S=1/2", 1)
    L = liouv_sites(s)
    op_z = OpSum()
    op_z += 1.0, "Sz", 1
    op_x = OpSum()
    op_x += 1.0, "Sx", 1
    obs_z = observable_measurement(op_z)
    obs_x = observable_measurement(op_x)
    tr_out = trace_out()
    prep = state_preparation(to_liouville(to_dm(MPS(s, ["Up"])); sites=L))

    prod1 = tr_out * prep
    prod2 = prep * tr_out
    @test prod1 isa ProductInstrument
    @test prod2 isa ProductInstrument
    @test prod1.input_instr === prep
    @test prod1.output_instr === tr_out
    @test prod2 == prod1

    same_leg = obs_z * obs_x
    @test same_leg isa SingleLegInstrument
    @test !(same_leg isa ProductInstrument)
    @test_throws ArgumentError prep * prep
    @test_throws MethodError obs_z * identity_operation()

    @test occursin("*", sprint(show, prod1))

    in1 = prime(L[1])
    out0 = L[1]

    measure_reprepare = obs_z * prep
    @test measure_reprepare isa ProductInstrument
    @test measure_reprepare.output_instr === obs_z
    @test measure_reprepare.input_instr === prep
    T_mr = instrument_itensor(measure_reprepare, [in1], [out0], 1)
    T_mr_ref = instrument_itensor(prep, [in1], 1) * instrument_itensor(obs_z, [out0], 0)
    @test isapprox(norm(T_mr - T_mr_ref), 0.0; atol=1e-12)

    T_prod = instrument_itensor(prod1, [in1], [out0], 1)
    T_ref = instrument_itensor(prep, [in1], 1) * instrument_itensor(tr_out, [out0], 0)
    @test isapprox(norm(T_prod - T_ref), 0.0; atol=1e-12)

    T_same = instrument_itensor(same_leg, [out0], 1)
    obs_h = apply(MPO(op_x, s), MPO(op_z, s))
    ref_l = to_liouville(obs_h; sites=L)
    T_same_ref = contract_core(ref_l.core)
    @test isapprox(norm(T_same - T_same_ref), 0.0; atol=1e-12)

    lr_left = left_action(op_z, s)
    lr_right = right_action(op_z, s)
    T_lr_left = instrument_itensor(lr_left, [in1], [out0], 1)
    T_lr_right = instrument_itensor(lr_right, [in1], [out0], 1)
    @test hasind(T_lr_left, in1) && hasind(T_lr_left, out0)
    @test hasind(T_lr_right, in1) && hasind(T_lr_right, out0)

    # General LeftRightOperator: ρ ↦ A ρ B with non-identity B (right action only on B side).
    O_mpo = MPO(op_z, s)
    Id_mpo = MPO(OpSum() + (1.0, "Id", 1), s)
    T_gen = instrument_itensor(left_right_operator(O_mpo, Id_mpo), [in1], [out0], 1)
    @test isapprox(Array(T_gen, in1, out0), Array(T_lr_left, in1, out0); atol=1e-12)

    iddef = identity_operation()
    seq = InstrumentSeq(iddef, 3)
    add!(seq, state_preparation(to_liouville(to_dm(MPS(s, ["Up"])); sites=L)), 0)
    add!(seq, prod1, 1)
    add!(seq, trace_out(), 3)
    _, _, missing_in, missing_out = _instrument_leg_maps(seq, 3)
    @test isempty(missing_in)
    @test isempty(missing_out)

    system = spin_system(s, OpSum() + (0.3, "Sz", 1))
    pt = build_process_tensor(system; dt=0.05, nsteps=3)
    rho0_h = to_dm(MPS(s, ["Up"]))
    seq_eval = InstrumentSeq(default=identity_operation(), nsteps=pt.nsteps)
    add!(seq_eval, state_preparation(rho0_h), 0)
    add!(seq_eval, observable_measurement(op_z), pt.nsteps)
    val = evaluate_process(pt, seq_eval)
    @test val isa ComplexF64
    @test isfinite(val)
end

@testset "Instruments.jl: LeftRightOperator and composed left_action" begin
    s = siteinds("S=1/2", 1)
    L = liouv_sites(s)
    op_z = OpSum()
    op_z += 1.0, "Sz", 1
    op_x = OpSum()
    op_x += 1.0, "Sx", 1
    in1 = prime(L[1])
    out0 = L[1]

    composed = observable_measurement(op_z) * observable_measurement(op_x)
    lr = left_action(composed, s)
    @test lr isa LeftRightOperator
    T_lr = instrument_itensor(lr, [in1], [out0], 1)
    @test norm(T_lr) > 0.0

    # Custom bilateral map: ρ ↦ S_- ρ S_+ (single-site example).
    Lop = OpSum()
    Lop += 1.0, "S-", 1
    L_mpo = MPO(Lop, s)
    Ldag_mpo = MPO(OpSum() + (1.0, "S+", 1), s)
    jump_lr = left_right_operator(L_mpo, Ldag_mpo)
    T_jump = instrument_itensor(jump_lr, [in1], [out0], 1)
    @test length(inds(T_jump)) == 2
    @test hasind(T_jump, in1) && hasind(T_jump, out0)
end

@testset "Instruments.jl: composed ObservableMeasurement * StatePreparation" begin
    s = siteinds("S=1/2", 1)
    L = liouv_sites(s)
    op_z = OpSum()
    op_z += 1.0, "Sz", 1
    rho_h = to_dm(MPS(s, ["Dn"]))
    prep_left = observable_measurement(op_z; leg_plev=1) * state_preparation(rho_h)
    T_left = instrument_itensor(prep_left, [prime(L[1])], 0)
    ref_left = instrument_itensor(
        state_preparation(apply(MPO(op_z, s), rho_h)),
        [prime(L[1])],
        0,
    )
    @test isapprox(norm(T_left - ref_left), 0.0; atol=1e-12)

    prep_right = state_preparation(rho_h) * observable_measurement(op_z; leg_plev=1)
    T_right = instrument_itensor(prep_right, [prime(L[1])], 0)
    ref_right = instrument_itensor(
        state_preparation(apply(rho_h, MPO(op_z, s))),
        [prime(L[1])],
        0,
    )
    @test isapprox(norm(T_right - ref_right), 0.0; atol=1e-12)
end

struct InstrumentsTestDummy <: AbstractInstrument end

@testset "Instruments.jl: instrument_itensor — unknown AbstractInstrument" begin
    # Given: user-defined AbstractInstrument with no instrument_itensor method.
    # When: instrument_itensor call.
    # Then: MethodError.
    @test_throws MethodError instrument_itensor(InstrumentsTestDummy(), Index[], 1)
end
