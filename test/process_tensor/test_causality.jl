# Causality tests for process tensors (embedded propagation only).
#
# At snapshot index `t` (1-based, matching `evolve` / `states_hilbert[t]`), the reduced
# system state should not depend on future interventions. Compare `evolve` with a reference
# schedule vs a schedule that inserts nontrivial two-leg instruments only after time `t`.

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

"""Default embedded schedule: prep at 0, IdentityOperation on bond slots."""
function _standard_seq(pt::ProcessTensor, rho0_h)
    seq = InstrumentSeq(default=IdentityOperation(), nsteps=pt.nsteps)
    add!(seq, StatePreparation(rho0_h), 0)
    return seq
end

"""
    _seq_future_causality_break(pt, rho0_h, open_at, future_probe)

Schedule identical to the reference up through evolve slot `open_at`, then apply
`future_instr` on all later bond slots and trace the terminal output.
"""
function _seq_future_causality_break(
    pt::ProcessTensor,
    rho0_h,
    open_at::Int,
    future_instr::AbstractInstrument,
)
    0 <= open_at < pt.nsteps || throw(BoundsError(0:(pt.nsteps - 1), open_at))
    seq = _standard_seq(pt, rho0_h)
    for step in (open_at + 1):(pt.nsteps - 1)
        add!(seq, future_instr, step)
    end
    if open_at < pt.nsteps - 1
        add!(seq, TraceOut(), pt.nsteps)
    end
    return seq
end

"""Physical Hilbert sites from a single-site `MPO{Hilbert}` density."""
function _phys_sites_from_rho0(rho0_h::AbstractMPO{Hilbert})
    return [only(filter(i -> plev(i) == 0, inds(rho0_h.core[j]))) for j in 1:length(rho0_h.core)]
end

"""
    _check_causality_triple(pt, rho0_h, t_idx; atol, order)

Compare reduced states at snapshot `t_idx` from reference evolution vs a future-perturbed schedule.
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
    default_instr = IdentityOperation()
    seq_std = _standard_seq(pt, rho0_h)
    O_probe = OpSum() + (1.0, "Sz", 1)
    seq_pert = _seq_future_causality_break(
        pt,
        rho0_h,
        open_at,
        left_action(O_probe, _phys_sites_from_rho0(rho0_h)),
    )

    ρ_ref = _mpo_to_dense(
        evolve(pt, seq_std; default_instr=default_instr, order=order).states_hilbert[t_idx],
    )
    ρ_pert = _mpo_to_dense(
        evolve(pt, seq_pert; default_instr=default_instr, order=order).states_hilbert[t_idx],
    )

    @test ρ_ref ≈ ρ_pert atol=atol
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
        )
        rho0_h = to_dm(MPS(s, ["Up"]))

        for t_idx in 2:nsteps
            _check_causality_triple(pt, rho0_h, t_idx; atol=1e-10)
        end
    end
end
