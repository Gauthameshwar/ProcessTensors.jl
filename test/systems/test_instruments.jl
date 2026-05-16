using ProcessTensors
using ITensors
using LinearAlgebra
using Test

if !(@isdefined hilbert_matrix_to_mpo)
    include(joinpath(@__DIR__, "..", "time_evolution", "tebd_test_utils.jl"))
end

const Ins = Base.getproperty(ProcessTensors, :Instruments)

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
    @test StatePreparation(to_liouville(to_dm(psi); sites=L)) isa StatePreparation
    @test StatePreparation(to_liouville(to_dm(psi); sites=L); leg_plev=0) isa StatePreparation
    @test StatePreparation(to_liouville(to_dm(psi))) isa StatePreparation

    @test TraceOut([L[1]]; leg_plev=0) isa TraceOut
    @test TraceOut([prime(L[1])]; leg_plev=1) isa TraceOut
    @test TraceOut() isa TraceOut

    @test ObservableMeasurement(op_z, [prime(L[1])]; leg_plev=1) isa ObservableMeasurement
    @test ObservableMeasurement(op_z, [L[1]]; leg_plev=0) isa ObservableMeasurement
    @test ObservableMeasurement(op_z) isa ObservableMeasurement

    # Given: wrong prime level on supplied sites vs declared leg_plev.
    # Then: constructor throws ArgumentError.
    @test_throws ArgumentError StatePreparation(psi, [L[1]]; leg_plev=1)

    s3 = siteinds("S=1/2", 3)
    H3 = OpSum(); H3 += 0.3, "Sz", 2
    # Given: lazy SystemPropagation / IdentityOperation (deferred PT legs).
    # Then: zero-length leg vectors are allowed.
    @test SystemPropagation(spin_system(s3, H3)) isa SystemPropagation
    @test IdentityOperation() isa IdentityOperation

    s1 = siteinds("S=1/2", 1)
    L1 = liouv_sites(s1)
    H1 = OpSum(); H1 += 0.5, "Sx", 1
    sys1 = spin_system(s1, H1)
    in1 = prime(L1[1])
    out0 = L1[1]
    prop_bound = SystemPropagation([in1], [out0], sys1)
    @test prop_bound.input_pt_sites == [in1]
    @test prop_bound.output_pt_sites == [out0]
    @test prop_bound.system === sys1
    id_bound = IdentityOperation([in1], [out0])
    @test id_bound.input_pt_sites == [in1]
    @test id_bound.output_pt_sites == [out0]
end

@testset "Instruments.jl: InstrumentSeq / resolve / add!" begin
    sys = spin_system(siteinds("S=1/2", 1), OpSum() + (0.2, "Sz", 1))
    def = SystemPropagation(sys)

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
    prep = StatePreparation(to_liouville(to_dm(MPS(s, ["Dn"])); sites=L))
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
    seq2 = InstrumentSeq(IdentityOperation(), 0)
    add!(seq2, def, 2)
    other = SystemPropagation(sys)
    add!(seq2, other, 2)
    @test resolve_instrument(seq2, 2) === other

    # Given: second schedule; add! at step 1.
    # When: resolve 1.
    # Then: returns the instrument passed to add!.
    seq3 = InstrumentSeq(IdentityOperation(), 0)
    add!(seq3, def, 1)
    @test resolve_instrument(seq3, 1) === def

    # Given: seq.nsteps bound and tstep above range.
    # When: add!.
    # Then: ArgumentError.
    seq4 = InstrumentSeq(IdentityOperation(), 3; entries=Dict{Int,AbstractInstrument}())
    @test_throws ArgumentError add!(seq4, def, 4)

    @test_throws ArgumentError resolve_instrument(seq2, -1)

    # resolve with fallback for k ≥ 1
    seq5 = InstrumentSeq(IdentityOperation(), 0)
    fb = SystemPropagation(sys)
    @test resolve_instrument(seq5, 1, fb) === fb
    add!(seq5, def, 1)
    @test resolve_instrument(seq5, 1, fb) === def

    seq_kw = InstrumentSeq(; default=IdentityOperation(), nsteps=2, entries=Dict(1 => def))
    @test resolve_instrument(seq_kw, 1) === def
    @test occursin("InstrumentSeq", sprint(show, seq_kw))

    seq_plus = InstrumentSeq(IdentityOperation(), 2)
    seq_plus += (def, 2)
    @test resolve_instrument(seq_plus, 2) === def
end

@testset "Instruments.jl: instrument_leg_maps coverage" begin
    sys = spin_system(siteinds("S=1/2", 1), OpSum() + (0.1, "Sz", 1))
    iddef = IdentityOperation()
    s_leg = siteinds("S=1/2", 1)
    L_leg = liouv_sites(s_leg)
    prep0 = StatePreparation(to_liouville(to_dm(MPS(s_leg, ["Up"])); sites=L_leg))

    # Given: nsteps=1, default two-leg, no prep at 0.
    # When: instrument_leg_maps.
    # Then: missing_in lists 0 (init leg), no missing_out (nsteps==1 expects no out coverage).
    seq = InstrumentSeq(iddef, 0)
    in_m, out_m, mi, mo = instrument_leg_maps(seq, 1)
    @test mi == [0]
    @test isempty(mo)

    # Given: same with explicit StatePreparation at 0.
    # When: maps.
    # Then: missing_in empty.
    s = siteinds("S=1/2", 1); L = liouv_sites(s)
    prep = StatePreparation(to_liouville(to_dm(MPS(s, ["Up"])); sites=L))
    seq_p = InstrumentSeq(iddef, 0; init=prep)
    _, _, mi2, mo2 = instrument_leg_maps(seq_p, 1)
    @test isempty(mi2) && isempty(mo2)

    # Given: nsteps ∈ {3,4}, default Identity plus init prep at 0.
    # When: maps.
    # Then: all required in/out slots covered; terminal open legs follow API contract.
    for n in (3, 4)
        seqn = InstrumentSeq(iddef, 0; init=prep0)
        _, _, minn, mout = instrument_leg_maps(seqn, n)
        @test isempty(minn) && isempty(mout)
    end

    # Given: sparse overrides — only step 3 set, rest default Identity, with prep at 0.
    # When: nsteps=4.
    # Then: in/out coverage complete.
    seq_s = InstrumentSeq(iddef, 0; init=prep0)
    add!(seq_s, SystemPropagation(sys), 3)
    _, _, mis, mos = instrument_leg_maps(seq_s, 4)
    @test isempty(mis) && isempty(mos)

    # Given: last-wins at same step changes TwoLeg instrument.
    # When: leg maps.
    # Then: maps reference the replaced instrument at that step.
    seq_l = InstrumentSeq(iddef, 0)
    spA = SystemPropagation(sys); spB = SystemPropagation(sys)
    add!(seq_l, spA, 2); add!(seq_l, spB, 2)
    im, om, _, _ = instrument_leg_maps(seq_l, 3)
    @test im[2] === spB
    @test om[1] === spB

    # Given: Observable on input leg (plev 1) at step 2 (replaces default two-leg there).
    # When: maps for nsteps=3.
    # Then: input slots stay covered via prep + other steps; output slot at tstep=1 lacks an instrument.
    op = OpSum(); op += 1.0, "Sz", 1
    seq_o = InstrumentSeq(iddef, 0; init=prep0)
    add!(seq_o, ObservableMeasurement(op, Index[]; leg_plev=1), 2)
    _, _, mio, moo = instrument_leg_maps(seq_o, 3)
    @test isempty(mio)
    @test moo == [1]

    # Given: no prep, Identity default, nsteps=3.
    # When: maps.
    # Then: missing_in reports 0 (documented sparse-init case).
    seq_np = InstrumentSeq(iddef, 0)
    _, _, mi_np, _ = instrument_leg_maps(seq_np, 3)
    @test 0 in mi_np
end

@testset "Instruments.jl: instrument_itensor — StatePreparation & dense data" begin
    s_spin = siteinds("S=1/2", 1)
    Ls = liouv_sites(s_spin)
    ρh = hilbert_matrix_to_mpo(randn(ComplexF64, 2, 2), s_spin)
    ρl = to_liouville(ρh; sites=Ls)
    prep = StatePreparation(ρl)
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
    prep_b = StatePreparation(ρbl)
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
        obs = ObservableMeasurement(os, Index[]; leg_plev=0)
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
    obs_n = ObservableMeasurement(on, Index[]; leg_plev=0)
    Ta = instrument_itensor(obs_n, [Lb[1]], 1)
    @test hasind(Ta, Lb[1])

    oad = OpSum(); oad += 1.0, "A", 1; oad += 1.0, "Adag", 1
    obs_ad = ObservableMeasurement(oad, Index[]; leg_plev=1)
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
    T = instrument_itensor(TraceOut(), [leg], 1)
    @test hastags(leg, "Liouv")
    data = vec(Array(T, leg))
    @test length(data) == d^2
    @test isapprox(data, vec_id)
end

@testset "Instruments.jl: instrument_itensor — SystemPropagation vs exp(dt * L)" begin
    s = siteinds("S=1/2", 1)
    L = liouv_sites(s)
    H = OpSum(); H += 0.6, "Sx", 1
    sys = spin_system(s, H)
    prop = SystemPropagation(sys)
    dt = 0.04
    in1 = prime(L[1])
    out0 = L[1]
    # Given: single-site Sy-type Hamiltonian and Liouvillian MPO.
    # When: instrument_itensor (order-1 Trotter) vs dense exp(dt * L).
    # Then: maps agree at requested indices.
    Tmap = instrument_itensor(prop, [in1], [out0], 1; dt=dt, order=1)
    L_mpo = MPO_Liouville(H, L; jump_ops=[])
    T_L = contract_core(L_mpo.core)
    d = dim(L[1])
    Lmat = reshape(Array(T_L, prime(L[1]), L[1]), d, d)
    Uexp = exp(dt * Lmat)
    A = reshape(Array(Tmap, in1, out0), d, d)
    @test isapprox(A, Uexp; atol=1e-10, rtol=1e-8)

    # Given: empty Hamiltonian (constructor emits a warning by design).
    # When: SystemPropagation ITensor.
    # Then: delta map between in/out legs; warning is captured/asserted by @test_logs.
    sys0 = @test_warn r"SpinSystem: H is empty" spin_system(s, OpSum())
    idprop = SystemPropagation(sys0)
    Tid = instrument_itensor(idprop, [in1], [out0], 1; dt=dt, order=1)
    @test isapprox(norm(Tid - delta(in1, out0)), 0.0; atol=1e-12)

    prop_bound = SystemPropagation([in1], [out0], sys)
    T_bound = instrument_itensor(prop_bound, Index[], Index[], 1; dt=dt, order=1)
    @test T_bound isa ITensor
    @test hasind(T_bound, in1) && hasind(T_bound, out0)
end

struct InstrumentsTestDummy <: AbstractInstrument end

@testset "Instruments.jl: instrument_itensor — unknown AbstractInstrument" begin
    # Given: user-defined AbstractInstrument with no instrument_itensor method.
    # When: instrument_itensor call.
    # Then: MethodError.
    @test_throws MethodError instrument_itensor(InstrumentsTestDummy(), Index[], 1)
end
