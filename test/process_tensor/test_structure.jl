# Fast structural / topological checks for process-tensor cores (no ED).
using ProcessTensors
using ITensors
using Test

if !isdefined(Main, :validate_process_tensor_structure)
    include(joinpath(@__DIR__, "pt_ed_test_utils.jl"))
end

function _markovian_system()
    s = siteinds("S=1/2", 1)
    H = OpSum() + (0.3, "Sz", 1)
    system = spin_system(s, H)
    return system, only(system.sites)
end

function _single_mode_bath(sys_site)
    env_phys = siteinds("S=1/2", 1)
    env_liouv = liouv_sites(env_phys)
    ρ_env = to_liouville(to_dm(MPS(env_phys, ["Up"])); sites=env_liouv)
    H_env = OpSum() + (0.5, "Sx", 1)
    cpl = OpSum() + (0.1, "Sz", 1, "Sz", 2)
    mode = spin_mode(env_liouv, H_env, ρ_env; coupling=cpl)
    return spin_bath([mode])
end

function _multimode_bath(sys_site)
    e1 = siteinds("S=1/2", 1)
    e2 = siteinds("S=1/2", 1)
    L1 = liouv_sites(e1)
    L2 = liouv_sites(e2)
    ρ1 = to_liouville(to_dm(MPS(e1, ["Up"])); sites=L1)
    ρ2 = to_liouville(to_dm(MPS(e2, ["Up"])); sites=L2)
    H_env = OpSum() + (1.0, "Sx", 1)
    m1 = spin_mode(L1, H_env, ρ1; coupling=OpSum() + (0.05, "Sz", 1, "Sz", 2))
    m2 = spin_mode(L2, H_env, ρ2; coupling=OpSum() + (0.03, "Sz", 1, "Sz", 2))
    return spin_bath([m1, m2])
end

function _build_pt_case(kind::Symbol, system, sys_site; nsteps::Int, embed::Bool)
    dt = 0.05
    if kind == :trivial
        return build_process_tensor(
            system;
            dt=dt,
            nsteps=nsteps,
            embed_system_propagation=embed,
        )
    elseif kind == :bathmode
        return build_process_tensor(
            system, sys_site;
            environment=_single_mode_bath(sys_site),
            dt=dt,
            nsteps=nsteps,
            embed_system_propagation=embed,
        )
    elseif kind == :multimode
        return build_process_tensor(
            system, sys_site;
            environment=_multimode_bath(sys_site),
            dt=dt,
            nsteps=nsteps,
            embed_system_propagation=embed,
        )
    else
        throw(ArgumentError("unknown build kind: $kind"))
    end
end

function _closed_seq(pt::ProcessTensor, rho0_h; embed::Bool)
    default = embed ? IdentityOperation() : SystemPropagation(pt.system)
    seq = InstrumentSeq(default=default, nsteps=pt.nsteps)
    add!(seq, StatePreparation(rho0_h), 0)
    add!(seq, TraceOut(), pt.nsteps)
    return seq
end

function _topology_closure_tensor(pt::ProcessTensor, seq::InstrumentSeq)
    prep = resolve_instrument(seq, 0)
    result = pt.core[1] * instrument_itensor(prep, input_sites(pt, 0), 0)
    for step in 1:(pt.nsteps - 1)
        instr = resolve_instrument(seq, step, seq.default)
        out_prev, in_curr = coupling_times(pt, step)
        bond = if instr isa TwoLegInstrument
            instrument_itensor(instr, in_curr, out_prev, step; dt=pt.dt)
        elseif instr isa SingleLegInstrument
            instr.leg_plev == 0 ?
                instrument_itensor(instr, out_prev, step) :
                instrument_itensor(instr, in_curr, step)
        else
            throw(ArgumentError("unsupported bond instrument $(typeof(instr)) at step=$step"))
        end
        result *= bond * pt.core[step + 1]
    end
    final_instr = resolve_instrument(seq, pt.nsteps, seq.default)
    if final_instr isa TraceOut || final_instr isa SingleLegInstrument
        out_prev, _ = coupling_times(pt, pt.nsteps)
        result *= instrument_itensor(final_instr, out_prev, pt.nsteps - 1)
    end
    return result
end

const _STRUCTURE_MATRIX = vcat(
    [(:trivial, n, true) for n in 2:5],
    [(:bathmode, n, true) for n in 2:5],
    [(:multimode, n, true) for n in 2:3],
    [(:bathmode, n, false) for n in 2:4],
)

    @testset "process_tensor: structure and topology" begin
    system, coupling_site = _markovian_system()
    rho0_h = to_dm(MPS(siteinds("S=1/2", 1), ["Up"]))

    @testset "validate_process_tensor_structure" begin
        for (kind, nsteps, embed) in _STRUCTURE_MATRIX
            @testset "$(kind), nsteps=$nsteps, embed=$embed" begin
                pt = _build_pt_case(kind, system, coupling_site; nsteps=nsteps, embed=embed)
                validate_process_tensor_structure(pt)
            end
        end
    end

    @testset "output_sites / input_sites fail closed on stray legs" begin
        pt = _build_pt_case(:bathmode, system, coupling_site; nsteps=4, embed=true)
        validate_process_tensor_structure(pt)
        for k in 0:3
            @test length(output_sites(pt, k)) == 1
            @test length(input_sites(pt, k)) == 1
            @test only(input_sites(pt, k)) in inds(pt.core[k + 1])
        end
    end

    @testset "topology closure (zero open indices)" begin
        for (kind, nsteps, embed) in _STRUCTURE_MATRIX
            @testset "$(kind), nsteps=$nsteps, embed=$embed" begin
                pt = _build_pt_case(kind, system, coupling_site; nsteps=nsteps, embed=embed)
                validate_process_tensor_structure(pt)
                seq = _closed_seq(pt, rho0_h; embed=embed)
                result = _topology_closure_tensor(pt, seq)
                @test length(inds(result)) == 0
                val = scalar(result)
                @test isapprox(real(val), 1.0; atol=1e-8)
            end
        end
    end

    @testset "evaluate_process agrees with manual topology closure (embed=true)" begin
        for (kind, nsteps, embed) in _STRUCTURE_MATRIX
            embed || continue
            @testset "$(kind), nsteps=$nsteps" begin
                pt = _build_pt_case(kind, system, coupling_site; nsteps=nsteps, embed=true)
                seq = _closed_seq(pt, rho0_h; embed=true)
                manual = scalar(_topology_closure_tensor(pt, seq))
                auto = evaluate_process(pt, seq)
                @test manual ≈ auto
            end
        end
    end
end
