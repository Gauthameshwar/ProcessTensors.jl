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
    """Physical sites `[system, bath_1, …, bath_M]` matching `kron(ρ_sys, ρ_b1, …)`."""
    _joint_phys_sites(sys_phys, env_phys) = Index[sys_phys[1], env_phys...]
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
        # Joint state is `kron(ρ_sys, ρ_env)`; system factor is the second tensor in kron.
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

if !isdefined(Main, :_evolve_joint_split_exact)
    """
    Exact dense reference for `evolve` split schedule at PT snapshot `k` (`t = k*dt`).

    Matches `states_liouville[k+1]`: slab `0` is one bath core; slabs `1:k` each add
    `SystemPropagation` then another bath core. Recomputes `exp(-im*dt*H)` every sub-step.
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
        for j in 0:k
            if j > 0
                U_sys = _exact_unitary_exp(H_sys, sys_phys, dt)
                rho = _apply_system_unitary_on_joint(rho, U_sys, denv)
            end
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

# Backward-compatible name: one exact bath step (matrix exponential, not Trotter).
if !isdefined(Main, :_dense_bath_step)
    function _dense_bath_step(nmodes::Int, mode_h_coeffs, mode_cpl_coeffs, dt::Real)
        joint_sites = siteinds("S=1/2", nmodes + 1)
        H_bg = _build_multimode_bath_opsum(nmodes, mode_h_coeffs, mode_cpl_coeffs)
        return _exact_unitary_exp(H_bg, joint_sites, dt)
    end
end
