# Causality tests for process tensors.
#
# At snapshot index `t` (1-based, matching `evolve` / `states_hilbert[t]`), the reduced
# system state should not depend on future interventions. We compare:
#   (1) `evolve` reference at `t`;
#   (2) `evaluate_process` with OpenOutput causality break plus IdentityOperation probes;
#   (3) same future break with SystemPropagation probes.
# All three must agree.

using ProcessTensors
using ITensors
using Test
using LinearAlgebra

if !isdefined(Main, :_mpo_to_dense)
    function _mpo_to_dense(mpo::AbstractMPO{Hilbert})
        sites = [only(filter(i -> plev(i) == 0, inds(mpo.core[j]))) for j in 1:length(mpo.core)]
        @assert length(sites) == 1 "Currently only supports single-site MPO"
        site = sites[1]
        d = dim(site)
        rho_dense = Array(mpo.core[1], prime(site), site)
        return reshape(ComplexF64.(rho_dense), d, d)
    end
end

"""Default open-system schedule: prep at 0, SystemPropagation on bond slots."""
function _standard_seq(pt::ProcessTensor, rho0_h)
    seq = InstrumentSeq(default=SystemPropagation(pt.system), nsteps=pt.nsteps)
    add!(seq, StatePreparation(rho0_h), 0)
    return seq
end

"""
    _seq_future_causality_break(pt, rho0_h, open_at, future_probe)

Construct a full contraction schedule that leaves `output_sites(pt, open_at)` open:
- at the first future step (`open_at + 1`) apply `OpenOutput()` (trace input, keep output);
- on later future steps apply `future_probe` to connect future output-input bonds;
- close the terminal output with `TraceOut` when `open_at < nsteps-1`.
"""
function _seq_future_causality_break(
    pt::ProcessTensor,
    rho0_h,
    open_at::Int,
    future_probe::AbstractInstrument,
)
    0 <= open_at < pt.nsteps || throw(BoundsError(0:(pt.nsteps - 1), open_at))
    seq = _standard_seq(pt, rho0_h)
    for step in (open_at + 1):(pt.nsteps - 1)
        if step == open_at + 1
            add!(seq, OpenOutput(), step)
        else
            add!(seq, future_probe, step)
        end
    end
    if open_at < pt.nsteps - 1
        add!(seq, TraceOut(), pt.nsteps)
    end
    return seq
end

"""
    _check_causality_triple(pt, rho0_h, t_idx; atol, order)

Return nothing.
"""
function _check_causality_triple(
    pt::ProcessTensor,
    rho0_h,
    t_idx::Int;
    atol=1e-10,
    order::Int=2,
)
    open_at = t_idx - 1
    0 <= open_at < pt.nsteps || throw(BoundsError(0:(pt.nsteps - 1), open_at))
    sysprop = SystemPropagation(pt.system)
    seq_std = _standard_seq(pt, rho0_h)
    seq_id = _seq_future_causality_break(pt, rho0_h, open_at, IdentityOperation())
    seq_sys = _seq_future_causality_break(pt, rho0_h, open_at, sysprop)

    # Trace out the system at the snapshot index without contracting the future cores
    ρ_trunc = _mpo_to_dense(
        evolve(pt, seq_std; default_instr=sysprop, order=order).states_hilbert[t_idx],
    )

    # Contract the full process tensor with IdentityOperation in the future
    ρ_id = _mpo_to_dense(
        to_hilbert(evaluate_process(pt, seq_id; default_instr=sysprop, order=order))
    )
    ρ_id /= tr(ρ_id)

    # Contract the full process tensor with SystemPropagation in the future
    ρ_sys = _mpo_to_dense(
        to_hilbert(evaluate_process(pt, seq_sys; default_instr=sysprop, order=order)),
    )
    ρ_sys /= tr(ρ_sys)

    @test ρ_trunc ≈ ρ_id atol=atol
    @test ρ_trunc ≈ ρ_sys atol=atol
    return nothing
end

@testset "process_tensor.jl: causality (marginal vs future padding)" begin
    @testset "single-mode PT: marginal at t vs future evaluate_process probes" begin
        s = siteinds("S=1/2", 1)
        e = siteinds("S=1/2", 1)
        L_env = liouv_sites(e)

        H_sys = OpSum()
        H_sys += 0.3, "Sz", 1
        system = spin_system(s, H_sys)

        ρ_env = to_liouville(to_dm(MPS(e, ["Up"])); sites=L_env)
        H_env = OpSum()
        H_env += 0.5, "Sx", 1
        cpl = OpSum() + (0.1, "Sz", 1, "Sz", 2)
        mode = SpinMode(L_env, H_env, ρ_env; coupling=cpl)
        bath = spin_bath([mode])

        dt = 0.05
        nsteps = 4
        pt = build_process_tensor(
            system,
            system.sites[1];
            environment=bath,
            dt=dt,
            nsteps=nsteps,
            embed_system_propagation=false,
        )
        rho0_h = to_dm(MPS(s, ["Up"]))

        for t_idx in 2:nsteps
            _check_causality_triple(pt, rho0_h, t_idx; atol=1e-10)
        end
    end

    @testset "multi-mode PT: marginal at t vs future evaluate_process probes" begin
        s = siteinds("S=1/2", 1)
        e1 = siteinds("S=1/2", 1)
        e2 = siteinds("S=1/2", 1)
        L_e1 = liouv_sites(e1)
        L_e2 = liouv_sites(e2)

        H_sys = OpSum()
        H_sys += 0.2, "Sz", 1
        system = spin_system(s, H_sys)

        ρ1 = to_liouville(to_dm(MPS(e1, ["Up"])); sites=L_e1)
        H_e1 = OpSum()
        H_e1 += 0.4, "Sx", 1
        cpl1 = OpSum() + (0.08, "Sz", 1, "Sz", 2)
        m1 = SpinMode(L_e1, H_e1, ρ1; coupling=cpl1)

        ρ2 = to_liouville(to_dm(MPS(e2, ["Up"])); sites=L_e2)
        H_e2 = OpSum()
        H_e2 += 0.6, "Sx", 1
        cpl2 = OpSum() + (0.07, "Sz", 1, "Sz", 2)
        m2 = SpinMode(L_e2, H_e2, ρ2; coupling=cpl2)

        bath = spin_bath([m1, m2])

        dt = 0.05
        nsteps = 3
        pt = build_process_tensor(
            system,
            system.sites[1];
            environment=bath,
            dt=dt,
            nsteps=nsteps,
            embed_system_propagation=false,
        )
        rho0_h = to_dm(MPS(s, ["Up"]))

        for t_idx in 2:nsteps
            _check_causality_triple(pt, rho0_h, t_idx; atol=1e-10)
        end
    end
end
