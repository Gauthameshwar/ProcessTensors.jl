# Shared helpers for process-tensor ED reference tests (included once via !isdefined guard).
using ProcessTensors
using ITensors
using LinearAlgebra

if !isdefined(Main, :_physical_sites_from_hilbert_mpo)
    function _physical_sites_from_hilbert_mpo(rho::AbstractMPO{Hilbert})
        return [only(filter(i -> plev(i) == 0, inds(rho.core[j]))) for j in 1:length(rho.core)]
    end
end

if !isdefined(Main, :_partial_trace_env)
    function _partial_trace_env(rho::AbstractMatrix{<:Number}, dsys::Int, denv::Int)
        rho4 = reshape(ComplexF64.(rho), dsys, denv, dsys, denv)
        out = zeros(ComplexF64, dsys, dsys)
        for e in 1:denv
            out .+= @view rho4[:, e, :, e]
        end
        return out
    end
end

if !isdefined(Main, :_joint_phys_sites)
    """
    Physical sites `[system, bath_1, …, bath_M]` for a joint `OpSum` / `MPO`.

    ITensor site order is system-first. Split-schedule dense states use
    `kron(ρ_bath, ρ_sys)` (environment left), matching [`_apply_system_unitary_on_joint`](@ref).
    """
    _joint_phys_sites(sys_phys, env_phys) = Index[sys_phys[1], env_phys...]
end

if !isdefined(Main, :_joint_hamiltonian_dense)
    """Dense joint Hamiltonian from an `OpSum` on `joint_sites` (via `MPO` → matrix)."""
    _joint_hamiltonian_dense(H::OpSum, joint_sites::AbstractVector{<:Index}) =
        dense_hamiltonian_matrix(H, joint_sites)
end

if !isdefined(Main, :_joint_unitary_exp)
    """`U = exp(-im * t * H)` on the joint Hilbert space defined by `joint_sites`."""
    _joint_unitary_exp(H::OpSum, joint_sites::AbstractVector{<:Index}, t::Real) =
        _exact_unitary_exp(H, joint_sites, t)
end

if !isdefined(Main, :_joint_density_B_at_0)
    """
    Joint density after inserting `B` on the system at `t = 0`, for split-schedule ED.

    Matches `StatePreparation(B * ρ_sys)` on the PT system leg with a product bath:
    `kron(ρ_env, B ρ_sys)`.
    """
    function _joint_density_B_at_0(
        rho_sys::AbstractMatrix{<:Number},
        rho_env::AbstractMatrix{<:Number},
        O_B::OpSum,
        sys_phys::AbstractVector{<:Index},
    )
        B = dense_hamiltonian_matrix(O_B, sys_phys)
        return kron(ComplexF64.(rho_env), B * ComplexF64.(rho_sys))
    end
end

if !isdefined(Main, :_exact_unitary_exp)
    """Exact `exp(-im * dt * H)` on Hilbert space (no Trotter factorisation)."""
    function _exact_unitary_exp(H::OpSum, sites::AbstractVector{<:Index}, dt::Real)
        H_dense = dense_hamiltonian_matrix(H, sites)
        return exp(-1im * float(dt) * ComplexF64.(Hermitian(H_dense)))
    end
end

if !isdefined(Main, :_apply_system_unitary_on_joint)
    function _apply_system_unitary_on_joint(
        rho_joint::AbstractMatrix{<:Number},
        U_sys::AbstractMatrix{<:Number},
        denv::Int,
    )
        # Split-schedule joint states use `kron(ρ_env, ρ_sys)`; system is the second factor.
        U_joint = kron(Matrix{ComplexF64}(I, denv, denv), ComplexF64.(U_sys))
        return U_joint * rho_joint * U_joint'
    end
end

if !isdefined(Main, :_build_joint_full_opsum)
    """Full joint Hamiltonian on `[system, bath_1, …]` = `H_sys` (site 1) + bath field + couplings."""
    _build_joint_full_opsum(H_sys::OpSum, H_bath::OpSum) = H_sys + H_bath
end

if !isdefined(Main, :_evolve_joint_full_exact)
    """
    Exact joint unitary dynamics: `ρ(t) = U(t) ρ(0) U(t)†` with `U(t) = exp(-im * t * H_full)`.
    Recomputes the matrix exponential at each call (no Trotter, no split sub-steps).
    """
    function _evolve_joint_full_exact(
        rho0::AbstractMatrix{<:Number},
        t::Real,
        H_full::OpSum,
        joint_sites::AbstractVector{<:Index},
    )
        t < 0 && throw(ArgumentError("_evolve_joint_full_exact: t must be non-negative; got $t."))
        iszero(t) && return ComplexF64.(rho0)
        U_t = _exact_unitary_exp(H_full, joint_sites, t)
        return U_t * ComplexF64.(rho0) * U_t'
    end
end

if !isdefined(Main, :_reduced_system_joint_full)
    function _reduced_system_joint_full(
        rho0::AbstractMatrix{<:Number},
        t::Real,
        H_full::OpSum,
        joint_sites::AbstractVector{<:Index},
        dsys::Int,
        denv::Int,
    )
        rho_t = _evolve_joint_full_exact(rho0, t, H_full, joint_sites)
        return _partial_trace_env(rho_t, dsys, denv)
    end
end

if !isdefined(Main, :_build_multimode_bath_opsum)
    """
    Bath + star coupling on joint sites `[system, bath_1, …, bath_M]`.
    Mode `m` lives on site `m+1`; couplings are between bath site `m+1` and system site `1`.
    """
    function _build_multimode_bath_opsum(
        nmodes::Int,
        mode_h_coeffs,
        mode_cpl_coeffs;
        h_op::AbstractString = "Sx",
        cpl_op::AbstractString = "Sz",
    )
        H = OpSum()
        for m in 1:nmodes
            H += mode_h_coeffs[m], h_op, m + 1
            H += mode_cpl_coeffs[m], cpl_op, m + 1, cpl_op, 1
        end
        return H
    end
end

if !isdefined(Main, :_PT_SPLIT_CORR_ATOL)
    # Residual PT (Liouville ITensor cores) vs dense split ED at moderate Δt.
    const _PT_SPLIT_CORR_ATOL = 2e-3
    const _PT_SPLIT_CORR_RTOL = 1e-2
end

if !isdefined(Main, :_evolve_joint_split_exact)
    """
    Exact dense reference for `evolve` split schedule at PT snapshot `k` (`t = k*dt`).

    Matches `states_liouville[k+1]`: each embedded PT slab applies `SystemPropagation`
    then a bath core (`exp(-im*dt*H)` recomputed every sub-step).
    """
    function _evolve_joint_split_exact(
        rho0::AbstractMatrix{<:Number},
        dt::Real,
        k::Int,
        H_sys::OpSum,
        H_bg::OpSum,
        sys_phys,
        joint_sites::AbstractVector{<:Index};
        denv::Int,
    )
        k >= 0 || throw(ArgumentError("_evolve_joint_split_exact: k must be non-negative; got $k."))
        rho = ComplexF64.(rho0)
        for _ in 0:k
            U_sys = _exact_unitary_exp(H_sys, sys_phys, dt)
            rho = _apply_system_unitary_on_joint(rho, U_sys, denv)
            U_bg = _exact_unitary_exp(H_bg, joint_sites, dt)
            rho = U_bg * rho * U_bg'
        end
        return rho
    end
end

if !isdefined(Main, :_joint_initial_density)
    function _joint_initial_density(sys_phys, env_phys)
        rho_joint = hilbert_mpo_to_dense(to_dm(MPS(sys_phys, ["Up"])), sys_phys)
        for m in 1:length(env_phys)
            rho_env = hilbert_mpo_to_dense(to_dm(MPS([env_phys[m]], ["Up"])), [env_phys[m]])
            rho_joint = kron(rho_joint, rho_env)
        end
        return rho_joint
    end
end

if !isdefined(Main, :_ed_expectation)
    """``Tr(ρ O)`` for a single-site (or joint) density matrix and `OpSum` observable."""
    function _ed_expectation(rho::AbstractMatrix{<:Number}, O::OpSum, sites::AbstractVector{<:Index})
        O_dense = dense_hamiltonian_matrix(O, sites)
        return real(tr(ComplexF64.(rho) * O_dense))
    end
end

if !isdefined(Main, :_schedule_default_instr_pt)
    function _schedule_default_instr_pt(::ProcessTensor)
        return IdentityOperation()
    end
end

if !isdefined(Main, :_seq_observable_terminal)
    """Fully contracted schedule measuring `O` on the terminal system output (time label `nsteps-1`)."""
    function _seq_observable_terminal(
        rho0_h,
        O::OpSum,
        nsteps::Int,
        default_instr::AbstractInstrument,
    )
        seq = InstrumentSeq(default=default_instr, nsteps=nsteps)
        add!(seq, StatePreparation(rho0_h), 0)
        add!(seq, ObservableMeasurement(O), nsteps)
        return seq
    end
end

if !isdefined(Main, :_seq_trace_terminal)
    function _seq_trace_terminal(rho0_h, nsteps::Int, default_instr::AbstractInstrument)
        seq = InstrumentSeq(default=default_instr, nsteps=nsteps)
        add!(seq, StatePreparation(rho0_h), 0)
        add!(seq, TraceOut(), nsteps)
        return seq
    end
end

if !isdefined(Main, :_seq_density_at_snapshot)
    """
    Schedule whose `evaluate_process` result is the system marginal at snapshot `k`
    (`t = k * dt`, output leg `out_k`). Uses `OpenOutput` for `k < nsteps-1` and the
    terminal-open schedule when `k == nsteps-1`.
    """
    function _seq_density_at_snapshot(
        rho0_h,
        k::Int,
        nsteps::Int,
        default_instr::AbstractInstrument,
    )
        0 <= k < nsteps || throw(ArgumentError("_seq_density_at_snapshot: k=$k out of range for nsteps=$nsteps."))
        seq = InstrumentSeq(default=default_instr, nsteps=nsteps)
        add!(seq, StatePreparation(rho0_h), 0)
        if k < nsteps - 1
            add!(seq, OpenOutput(), k + 1)
            for step in (k + 2):(nsteps - 1)
                add!(seq, IdentityOperation(), step)
            end
            add!(seq, TraceOut(), nsteps)
        end
        return seq
    end
end

if !isdefined(Main, :_hilbert_mpo_to_dense_one_site)
    function _hilbert_mpo_to_dense_one_site(rho_h::AbstractMPO{Hilbert})
        return hilbert_mpo_to_dense(rho_h, _physical_sites_from_hilbert_mpo(rho_h))
    end
end

if !isdefined(Main, :_apply_sys_op_joint)
    """
    Apply a system `OpSum` on the joint density `kron(ρ_env, ρ_sys)` (system is the second factor).

    - `side = :left`: ``\\mathcal{L}_O[\\rho] = O\\rho`` (joint map `I_env ⊗ O`).
    - `side = :right`: ``\\mathcal{R}_O[\\rho] = \\rho O`` (joint map `I_env ⊗ O` on the right).
    """
    function _apply_sys_op_joint(
        rho_joint::AbstractMatrix{<:Number},
        O::OpSum,
        sys_phys::AbstractVector{<:Index},
        denv::Int;
        side::Symbol,
    )
        Omat = dense_hamiltonian_matrix(O, sys_phys)
        O_joint = kron(Matrix{ComplexF64}(I, denv, denv), ComplexF64.(Omat))
        ρ = ComplexF64.(rho_joint)
        return side === :left ? O_joint * ρ : ρ * O_joint
    end
end

if !isdefined(Main, :_evolve_joint_split_continue)
    """Apply `n_steps` split (system then bath) updates from an intermediate joint state."""
    function _evolve_joint_split_continue(
        rho_joint::AbstractMatrix{<:Number},
        dt::Real,
        n_steps::Int,
        H_sys::OpSum,
        H_bg::OpSum,
        sys_phys,
        joint_sites::AbstractVector{<:Index};
        denv::Int,
    )
        n_steps >= 0 || throw(ArgumentError("_evolve_joint_split_continue: n_steps must be ≥ 0."))
        ρ = ComplexF64.(rho_joint)
        for _ in 1:n_steps
            U_sys = _exact_unitary_exp(H_sys, sys_phys, dt)
            ρ = _apply_system_unitary_on_joint(ρ, U_sys, denv)
            U_bg = _exact_unitary_exp(H_bg, joint_sites, dt)
            ρ = U_bg * ρ * U_bg'
        end
        return ρ
    end
end

if !isdefined(Main, :_ed_corr_two_time)
    """
    ``\\langle A(t_A)\\, B(t_B) \\rangle`` from split joint ED (evolves only to ``t_{\\mathrm{late}}``).

    Time indices `n_A`, `n_B` match [`two_time_correlation_seq`](@ref): snapshot after `n` split steps.
  Independent of any extra PT length beyond ``n_{\\mathrm{late}}``.
    """
    function _ed_corr_two_time(
        rho_sys0_h,
        rho_env0_h,
        O_A::OpSum,
        O_B::OpSum,
        n_A::Int,
        n_B::Int,
        dt::Real,
        H_sys::OpSum,
        H_bg::OpSum,
        sys_phys::AbstractVector{<:Index},
        env_phys::AbstractVector{<:Index};
        denv::Int = dim(only(env_phys)),
    )
        rho_sys = hilbert_mpo_to_dense(rho_sys0_h, sys_phys)
        rho_env = hilbert_mpo_to_dense(rho_env0_h, env_phys)
        joint_sites = _joint_phys_sites(sys_phys, env_phys)
        dsys = dim(only(sys_phys))

        n_late = max(n_A, n_B)
        n_early = min(n_A, n_B)

        if n_A == n_B
            ρ_joint = kron(ComplexF64.(rho_env), ComplexF64.(rho_sys))
            ρ_joint = _evolve_joint_split_exact(
                ρ_joint,
                dt,
                n_late,
                H_sys,
                H_bg,
                sys_phys,
                joint_sites;
                denv=denv,
            )
            A = dense_hamiltonian_matrix(O_A, sys_phys)
            B = dense_hamiltonian_matrix(O_B, sys_phys)
            ρ_sys_t = _partial_trace_env(ρ_joint, dsys, denv)
            return tr(A * B * ρ_sys_t)
        end

        if n_A > n_B
            O_early, O_late = O_B, O_A
            early_side = :left
        else
            O_early, O_late = O_A, O_B
            early_side = :right
        end

        if n_early == 0
            if early_side === :left
                ρ_joint = _joint_density_B_at_0(rho_sys, rho_env, O_early, sys_phys)
            else
                Omat = dense_hamiltonian_matrix(O_early, sys_phys)
                ρ_joint = kron(ComplexF64.(rho_env), ComplexF64.(rho_sys) * Omat)
            end
            ρ_joint = _evolve_joint_split_exact(
                ρ_joint,
                dt,
                n_late,
                H_sys,
                H_bg,
                sys_phys,
                joint_sites;
                denv=denv,
            )
        else
            ρ_joint = kron(ComplexF64.(rho_env), ComplexF64.(rho_sys))
            ρ_joint = _evolve_joint_split_exact(
                ρ_joint,
                dt,
                n_early,
                H_sys,
                H_bg,
                sys_phys,
                joint_sites;
                denv=denv,
            )
            ρ_joint = _apply_sys_op_joint(ρ_joint, O_early, sys_phys, denv; side=early_side)
            ρ_joint = _evolve_joint_split_continue(
                ρ_joint,
                dt,
                n_late - n_early,
                H_sys,
                H_bg,
                sys_phys,
                joint_sites;
                denv=denv,
            )
        end

        O_late = dense_hamiltonian_matrix(O_late, sys_phys)
        ρ_sys_t = _partial_trace_env(ρ_joint, dsys, denv)
        return tr(O_late * ρ_sys_t)
    end
end

# Backward-compatible name: one exact bath step (matrix exponential, not Trotter).
if !isdefined(Main, :_dense_bath_step)
    function _dense_bath_step(nmodes::Int, mode_h_coeffs, mode_cpl_coeffs, dt::Real)
        joint_sites = siteinds("S=1/2", nmodes + 1)
        H_bg = _build_multimode_bath_opsum(nmodes, mode_h_coeffs, mode_cpl_coeffs)
        return _exact_unitary_exp(H_bg, joint_sites, dt)
    end
end

if !isdefined(Main, :validate_process_tensor_structure)
    function validate_process_tensor_structure(pt::ProcessTensor)
        length(pt.core) == pt.nsteps || throw(
            ArgumentError(
                "validate_process_tensor_structure: expected $(pt.nsteps) cores, got $(length(pt.core)).",
            ),
        )
        nmodes = pt.environment === nothing ? 0 : length(pt.environment.modes)
        for k in 0:(pt.nsteps - 1)
            core_k = pt.core[k + 1]
            sys_legs = Index[idx for idx in inds(core_k) if !has_tag_token(idx, "Link")]
            link_legs = Index[idx for idx in inds(core_k) if has_tag_token(idx, "Link")]

            length(sys_legs) == 2 || throw(
                ArgumentError(
                    "validate_process_tensor_structure: core k=$k expected 2 system legs, " *
                    "found $(length(sys_legs)).",
                ),
            )
            for idx in sys_legs
                plev(idx) in (0, 1) || throw(
                    ArgumentError(
                        "validate_process_tensor_structure: core k=$k has system leg with plev=$(plev(idx)); " *
                        "expected 0 or 1 (no leaked internal prime levels).",
                    ),
                )
                tstep_str = tag_value(idx, "tstep=")
                tstep_str === nothing && throw(
                    ArgumentError("validate_process_tensor_structure: core k=$k system leg missing tstep= tag."),
                )
                parse(Int, tstep_str) == k || throw(
                    ArgumentError(
                        "validate_process_tensor_structure: core k=$k has system leg at tstep=$tstep_str; expected $k.",
                    ),
                )
            end

            out = only(output_sites(pt, k))
            inn = only(input_sites(pt, k))
            prime(out) == inn || throw(
                ArgumentError(
                    "validate_process_tensor_structure: core k=$k input/output legs are not a prime pair.",
                ),
            )

            if nmodes == 0
                isempty(link_legs) || throw(
                    ArgumentError(
                        "validate_process_tensor_structure: Markovian core k=$k must have no Link legs; " *
                        "found $(length(link_legs)).",
                    ),
                )
            elseif pt.nsteps == 1
                isempty(link_legs) || throw(
                    ArgumentError(
                        "validate_process_tensor_structure: single-slab bath PT must have bath links contracted; " *
                        "core k=$k still has $(length(link_legs)) Link leg(s).",
                    ),
                )
            elseif k == 0
                length(link_legs) == 1 || throw(
                    ArgumentError(
                        "validate_process_tensor_structure: first bath core k=0 expected 1 Link leg, " *
                        "found $(length(link_legs)).",
                    ),
                )
            elseif k == pt.nsteps - 1
                length(link_legs) == 1 || throw(
                    ArgumentError(
                        "validate_process_tensor_structure: last bath core k=$k expected 1 Link leg, " *
                        "found $(length(link_legs)).",
                    ),
                )
            else
                length(link_legs) == 2 || throw(
                    ArgumentError(
                        "validate_process_tensor_structure: interior bath core k=$k expected 2 Link legs, " *
                        "found $(length(link_legs)).",
                    ),
                )
            end
        end
        return nothing
    end
end
