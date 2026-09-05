# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# Compare Dense PT against Hilbert ED, Liouville ED, and Strang-split ED.

include(joinpath(@__DIR__, "common.jl"))

using ITensors.Ops: Exact, Trotter

const _AGENT_LOG = "/home/gautham/ProcessTensors.jl/.cursor/debug-a87e71.log"

function _agent_log(hypothesis_id, location, message, data)
    # #region agent log
    open(_AGENT_LOG, "a") do io
        print(io, "{\"sessionId\":\"a87e71\",\"runId\":\"mismatch4\",\"hypothesisId\":\"", hypothesis_id)
        print(io, "\",\"location\":\"", location, "\",\"message\":\"", message, "\",\"timestamp\":")
        print(io, round(Int, time() * 1000), ",\"data\":{")
        first_item = true
        for (key, value) in data
            first_item || print(io, ",")
            first_item = false
            print(io, "\"", key, "\":")
            if value isa AbstractFloat && isfinite(value)
                print(io, value)
            elseif value isa Integer
                print(io, value)
            else
                print(io, '"', value, '"')
            end
        end
        println(io, "}}")
    end
    # #endregion
    return nothing
end

function pauli_from_rho(ρ, Sx, Sy, Sz)
    return (
        sx=real(tr(ρ * Sx)),
        sy=real(tr(ρ * Sy)),
        sz=real(tr(ρ * Sz)),
    )
end

function shared_spin_model()
    sys_phys = siteinds("S=1/2", 1)
    env_phys = siteinds("S=1/2", 3)
    env_liouv = liouv_sites(env_phys)
    H_sys = OpSum() + (OMEGA_SYSTEM, "Sx", 1)
    system = spin_system(sys_phys, H_sys)
    modes = SpinMode[]
    for (m, (ω, g, axis)) in enumerate(zip(MODE_FREQUENCIES, MODE_COUPLINGS, MODE_AXES))
        ρ_mode = to_liouville(to_dm(MPS([env_phys[m]], ["Up"])); sites=[env_liouv[m]])
        H_mode = OpSum() + (ω, "Sx", 1)
        coupling = OpSum() + (g, axis, 1, axis, 2)
        push!(modes, spin_mode([env_liouv[m]], H_mode, ρ_mode; coupling=coupling))
    end
    bath = spin_bath(modes)
    joint_phys = Index[sys_phys[1], env_phys...]
    joint_liouv = liouv_sites(joint_phys)
    H_full = OpSum() + (OMEGA_SYSTEM, "Sx", 1)
    H_bath = OpSum()
    for (m, (ω, g, axis)) in enumerate(zip(MODE_FREQUENCIES, MODE_COUPLINGS, MODE_AXES))
        H_full += ω, "Sx", m + 1
        H_full += g, axis, m + 1, axis, 1
        H_bath += ω, "Sx", m + 1
        H_bath += g, axis, m + 1, axis, 1
    end
    ρ_sys0 = to_dm(MPS(sys_phys, ["Up"]))
    ρ_joint0_h = to_dm(MPS(joint_phys, fill("Up", 4)))
    ρ_joint0_l = to_liouville(ρ_joint0_h; sites=joint_liouv)
    ρ_joint0_d = hilbert_mpo_to_dense(ρ_joint0_h, joint_phys)
    one = sys_phys
    Sx = dense_hamiltonian_matrix(OpSum() + (1.0, "Sx", 1), one)
    Sy = dense_hamiltonian_matrix(OpSum() + (1.0, "Sy", 1), one)
    Sz = dense_hamiltonian_matrix(OpSum() + (1.0, "Sz", 1), one)
    Sx_joint = dense_hamiltonian_matrix(OpSum() + (1.0, "Sx", 1), joint_phys)
    return (;
        system, bath, sys_phys, env_phys, joint_phys, joint_liouv, H_full, H_sys, H_bath,
        ρ_sys0, ρ_joint0_l, ρ_joint0_d, Sx, Sy, Sz, Sx_joint,
    )
end

function hilbert_ed_reduced(model, t)
    H = Hermitian(dense_hamiltonian_matrix(model.H_full, model.joint_phys))
    decomposition = eigen(H)
    phases = exp.(-1im * float(t) .* decomposition.values)
    U = decomposition.vectors * Diagonal(phases) * decomposition.vectors'
    ρ_joint = U * model.ρ_joint0_d * U'
    ρ_red = partial_trace_environment(ρ_joint, 2, 8)
    joint_sx = real(tr(ρ_joint * model.Sx_joint))
    red_sx = real(tr(ρ_red * model.Sx))
    return ρ_red, joint_sx, red_sx
end

function liouville_ed_reduced(model, t)
    U_L = liouvillian_propagator(model.H_full, model.joint_liouv, t; alg=Exact())
    ρ_l = apply(U_L, copy(model.ρ_joint0_l); cutoff=0.0, maxdim=typemax(Int))
    ρ_h = to_hilbert(ρ_l)
    sites = [
        only(filter(i -> plev(i) == 0 && hastags(i, "Site"), inds(ρ_h.core[j])))
        for j in eachindex(ρ_h.core)
    ]
    T = foldl(*, ρ_h)
    A = Array(T, prime.(sites)..., sites...)
    ρ4 = reshape(ComplexF64.(A), 2, 8, 2, 8)
    ρ_red = zeros(ComplexF64, 2, 2)
    for e in 1:8
        ρ_red .+= @view ρ4[:, e, :, e]
    end
    return ρ_red
end

function strang_split_ed(model, dt, nsteps)
    Hs = dense_hamiltonian_matrix(model.H_sys, model.sys_phys)
    Hb = dense_hamiltonian_matrix(model.H_bath, model.joint_phys)
    Us_half = exp(-1im * (dt / 2) * ComplexF64.(Hermitian(Hs)))
    Ub = exp(-1im * dt * ComplexF64.(Hermitian(Hb)))
    Ienv = Matrix{ComplexF64}(I, 8, 8)
    # ITensor site 1 (system) is the fastest reshape index, i.e. the last Kronecker factor.
    Us_half_j = kron(Ienv, Us_half)
    ρ = copy(model.ρ_joint0_d)
    reduced = Vector{Matrix{ComplexF64}}(undef, nsteps)
    for k in 1:nsteps
        ρ = Us_half_j * ρ * Us_half_j'
        ρ = Ub * ρ * Ub'
        ρ = Us_half_j * ρ * Us_half_j'
        reduced[k] = partial_trace_environment(ρ, 2, 8)
    end
    return reduced
end

function hilbert_bath_only_reduced(model, t)
    H = Hermitian(dense_hamiltonian_matrix(model.H_bath, model.joint_phys))
    decomposition = eigen(H)
    phases = exp.(-1im * float(t) .* decomposition.values)
    U = decomposition.vectors * Diagonal(phases) * decomposition.vectors'
    ρ_joint = U * model.ρ_joint0_d * U'
    return partial_trace_environment(ρ_joint, 2, 8)
end

function liouville_bath_only_reduced(model, t)
    U_L = liouvillian_propagator(model.H_bath, model.joint_liouv, t; alg=Exact())
    ρ_l = apply(U_L, copy(model.ρ_joint0_l); cutoff=0.0, maxdim=typemax(Int))
    ρ_h = to_hilbert(ρ_l)
    sites = [
        only(filter(i -> plev(i) == 0 && hastags(i, "Site"), inds(ρ_h.core[j])))
        for j in eachindex(ρ_h.core)
    ]
    T = foldl(*, ρ_h)
    A = Array(T, prime.(sites)..., sites...)
    ρ4 = reshape(ComplexF64.(A), 2, 8, 2, 8)
    ρ_red = zeros(ComplexF64, 2, 2)
    for e in 1:8
        ρ_red .+= @view ρ4[:, e, :, e]
    end
    return ρ_red
end

function sz_coupled_spin_model()
    sys_phys = siteinds("S=1/2", 1)
    env_phys = siteinds("S=1/2", 3)
    env_liouv = liouv_sites(env_phys)
    H_sys = OpSum() + (OMEGA_SYSTEM, "Sx", 1)
    system = spin_system(sys_phys, H_sys)
    modes = SpinMode[]
    for (m, (ω, g)) in enumerate(zip(MODE_FREQUENCIES, MODE_COUPLINGS))
        ρ_mode = to_liouville(to_dm(MPS([env_phys[m]], ["Up"])); sites=[env_liouv[m]])
        H_mode = OpSum() + (ω, "Sx", 1)
        coupling = OpSum() + (g, "Sz", 1, "Sz", 2)
        push!(modes, spin_mode([env_liouv[m]], H_mode, ρ_mode; coupling=coupling))
    end
    bath = spin_bath(modes)
    joint_phys = Index[sys_phys[1], env_phys...]
    joint_liouv = liouv_sites(joint_phys)
    H_full = OpSum() + (OMEGA_SYSTEM, "Sx", 1)
    H_bath = OpSum()
    for (m, (ω, g)) in enumerate(zip(MODE_FREQUENCIES, MODE_COUPLINGS))
        H_full += ω, "Sx", m + 1
        H_full += g, "Sz", m + 1, "Sz", 1
        H_bath += ω, "Sx", m + 1
        H_bath += g, "Sz", m + 1, "Sz", 1
    end
    ρ_sys0 = to_dm(MPS(sys_phys, ["Up"]))
    ρ_joint0_h = to_dm(MPS(joint_phys, fill("Up", 4)))
    ρ_joint0_l = to_liouville(ρ_joint0_h; sites=joint_liouv)
    ρ_joint0_d = hilbert_mpo_to_dense(ρ_joint0_h, joint_phys)
    Sx = dense_hamiltonian_matrix(OpSum() + (1.0, "Sx", 1), sys_phys)
    Sy = dense_hamiltonian_matrix(OpSum() + (1.0, "Sy", 1), sys_phys)
    Sz = dense_hamiltonian_matrix(OpSum() + (1.0, "Sz", 1), sys_phys)
    Sx_joint = dense_hamiltonian_matrix(OpSum() + (1.0, "Sx", 1), joint_phys)
    return (;
        system, bath, sys_phys, env_phys, joint_phys, joint_liouv, H_full, H_sys, H_bath,
        ρ_sys0, ρ_joint0_l, ρ_joint0_d, Sx, Sy, Sz, Sx_joint,
    )
end

function main()
    dt = 0.1
    nsteps = 6
    model = shared_spin_model()
    ρ_h0, j0, r0 = hilbert_ed_reduced(model, 0.0)
    _agent_log("A", "debug_ed_vs_pt.jl:joint_vs_red", "ED joint Sx vs reduced Sx at t=0", Dict(
        "joint_sx" => j0, "reduced_sx" => r0, "absdiff" => abs(j0 - r0),
    ))

    t_end = nsteps * dt
    pt_iso = build_process_tensor(
        model.system; method=Dense(), dt=dt, nsteps=nsteps, alg=Exact(), progress=false,
    )
    traj_iso = evolve(pt_iso, model.ρ_sys0; progress=false)
    ρ_iso = one_site_density_matrix(traj_iso.states_hilbert[end])
    iso_obs = pauli_from_rho(ρ_iso, model.Sx, model.Sy, model.Sz)
    ρ0 = one_site_density_matrix(model.ρ_sys0)
    U_iso = exp(-1im * t_end * ComplexF64.(Hermitian(dense_hamiltonian_matrix(model.H_sys, model.sys_phys))))
    ed_iso = pauli_from_rho(U_iso * ρ0 * U_iso', model.Sx, model.Sy, model.Sz)
    _agent_log("F", "debug_ed_vs_pt.jl:isolated", "Isolated PT vs Hilbert Sy/Sz at T", Dict(
        "t" => t_end,
        "pt_sx" => iso_obs.sx, "pt_sy" => iso_obs.sy, "pt_sz" => iso_obs.sz,
        "ed_sx" => ed_iso.sx, "ed_sy" => ed_iso.sy, "ed_sz" => ed_iso.sz,
        "d_sy" => iso_obs.sy - ed_iso.sy,
        "d_sz" => iso_obs.sz - ed_iso.sz,
        "analytic_sy" => -0.5 * sin(OMEGA_SYSTEM * t_end),
        "analytic_sz" => 0.5 * cos(OMEGA_SYSTEM * t_end),
    ))

    sys_neg = spin_system(model.sys_phys, OpSum() + (-OMEGA_SYSTEM, "Sx", 1))
    pt_neg = build_process_tensor(
        sys_neg; method=Dense(), environment=model.bath, dt=dt, nsteps=nsteps,
        alg=Exact(), sys_alg=Trotter{2}(), progress=false,
    )
    traj_neg = evolve(pt_neg, model.ρ_sys0; progress=false)
    neg_obs = pauli_from_rho(
        one_site_density_matrix(traj_neg.states_hilbert[end]), model.Sx, model.Sy, model.Sz,
    )
    ρ_h_end, _, _ = hilbert_ed_reduced(model, t_end)
    h_end = pauli_from_rho(ρ_h_end, model.Sx, model.Sy, model.Sz)
    _agent_log("G", "debug_ed_vs_pt.jl:neg_Hs", "PT with -H_S vs original Hilbert ED at T", Dict(
        "t" => t_end,
        "neg_sx" => neg_obs.sx, "neg_sy" => neg_obs.sy, "neg_sz" => neg_obs.sz,
        "hilbert_sx" => h_end.sx, "hilbert_sy" => h_end.sy, "hilbert_sz" => h_end.sz,
        "abs_d_sx" => abs(neg_obs.sx - h_end.sx),
        "abs_d_sy" => abs(neg_obs.sy - h_end.sy),
        "abs_d_sz" => abs(neg_obs.sz - h_end.sz),
    ))

    sys0 = spin_system(model.sys_phys, OpSum())
    pt_bath = build_process_tensor(
        sys0; method=Dense(), environment=model.bath, dt=dt, nsteps=nsteps,
        alg=Exact(), sys_alg=Trotter{2}(), progress=false,
    )
    traj_bath = evolve(pt_bath, model.ρ_sys0; progress=false)
    bath_pt = pauli_from_rho(
        one_site_density_matrix(traj_bath.states_hilbert[end]), model.Sx, model.Sy, model.Sz,
    )
    bath_h = pauli_from_rho(hilbert_bath_only_reduced(model, t_end), model.Sx, model.Sy, model.Sz)
    bath_l = pauli_from_rho(liouville_bath_only_reduced(model, t_end), model.Sx, model.Sy, model.Sz)
    _agent_log("H", "debug_ed_vs_pt.jl:bath_only_pt", "H_S=0 PT vs Hilbert/Liouville H_bath", Dict(
        "t" => t_end,
        "pt_sx" => bath_pt.sx, "pt_sy" => bath_pt.sy, "pt_sz" => bath_pt.sz,
        "hilbert_sx" => bath_h.sx, "hilbert_sy" => bath_h.sy, "hilbert_sz" => bath_h.sz,
        "d_sx" => bath_pt.sx - bath_h.sx,
        "d_sy" => bath_pt.sy - bath_h.sy,
        "d_sz" => bath_pt.sz - bath_h.sz,
    ))
    _agent_log("I", "debug_ed_vs_pt.jl:bath_only_liouv", "Joint Liouville H_bath vs Hilbert H_bath", Dict(
        "t" => t_end,
        "liouv_sx" => bath_l.sx, "liouv_sy" => bath_l.sy, "liouv_sz" => bath_l.sz,
        "hilbert_sx" => bath_h.sx, "hilbert_sy" => bath_h.sy, "hilbert_sz" => bath_h.sz,
        "d_sx" => bath_l.sx - bath_h.sx,
        "d_sy" => bath_l.sy - bath_h.sy,
        "d_sz" => bath_l.sz - bath_h.sz,
    ))

    model_sz = sz_coupled_spin_model()
    pt_sz = build_process_tensor(
        model_sz.system; method=Dense(), environment=model_sz.bath, dt=dt, nsteps=nsteps,
        alg=Exact(), sys_alg=Trotter{2}(), progress=false,
    )
    traj_sz = evolve(pt_sz, model_sz.ρ_sys0; progress=false)
    sz_pt = pauli_from_rho(
        one_site_density_matrix(traj_sz.states_hilbert[end]), model_sz.Sx, model_sz.Sy, model_sz.Sz,
    )
    ρ_sz_h, _, _ = hilbert_ed_reduced(model_sz, t_end)
    sz_h = pauli_from_rho(ρ_sz_h, model_sz.Sx, model_sz.Sy, model_sz.Sz)
    _agent_log("K", "debug_ed_vs_pt.jl:sz_couplings", "All-Sz couplings PT vs Hilbert", Dict(
        "t" => t_end,
        "pt_sx" => sz_pt.sx, "pt_sy" => sz_pt.sy, "pt_sz" => sz_pt.sz,
        "hilbert_sx" => sz_h.sx, "hilbert_sy" => sz_h.sy, "hilbert_sz" => sz_h.sz,
        "d_sx" => sz_pt.sx - sz_h.sx,
        "d_sy" => sz_pt.sy - sz_h.sy,
        "d_sz" => sz_pt.sz - sz_h.sz,
    ))

    pt = build_process_tensor(
        model.system;
        method=Dense(),
        environment=model.bath,
        dt=dt,
        nsteps=nsteps,
        alg=Exact(),
        sys_alg=Trotter{2}(),
        progress=false,
    )
    traj = evolve(pt, model.ρ_sys0; progress=false)
    split_rhos = strang_split_ed(model, dt, nsteps)

    max_pt_hilbert = 0.0
    max_pt_liouv = 0.0
    max_pt_split = 0.0
    max_hilbert_liouv = 0.0
    max_sz_pt_hilbert = 0.0
    max_dt_independent = 0.0

    for k in 1:nsteps
        t = k * dt
        ρ_pt = one_site_density_matrix(traj.states_hilbert[k])
        pt_obs = pauli_from_rho(ρ_pt, model.Sx, model.Sy, model.Sz)
        ρ_h, joint_sx, red_sx = hilbert_ed_reduced(model, t)
        h_obs = pauli_from_rho(ρ_h, model.Sx, model.Sy, model.Sz)
        ρ_l = liouville_ed_reduced(model, t)
        l_obs = pauli_from_rho(ρ_l, model.Sx, model.Sy, model.Sz)
        s_obs = pauli_from_rho(split_rhos[k], model.Sx, model.Sy, model.Sz)
        max_pt_hilbert = max(max_pt_hilbert, abs(pt_obs.sx - h_obs.sx))
        max_pt_liouv = max(max_pt_liouv, abs(pt_obs.sx - l_obs.sx))
        max_pt_split = max(max_pt_split, abs(pt_obs.sx - s_obs.sx))
        max_hilbert_liouv = max(max_hilbert_liouv, abs(h_obs.sx - l_obs.sx), abs(joint_sx - red_sx))
        max_sz_pt_hilbert = max(max_sz_pt_hilbert, abs(pt_obs.sz - h_obs.sz))
        if k == nsteps
            _agent_log("B", "debug_ed_vs_pt.jl:observables", "Last-step Sx Sy Sz", Dict(
                "t" => t,
                "pt_sx" => pt_obs.sx, "pt_sy" => pt_obs.sy, "pt_sz" => pt_obs.sz,
                "hilbert_sx" => h_obs.sx, "hilbert_sy" => h_obs.sy, "hilbert_sz" => h_obs.sz,
                "liouv_sx" => l_obs.sx, "split_sx" => s_obs.sx,
                "joint_sx" => joint_sx, "red_sx" => red_sx,
            ))
            _agent_log("J", "debug_ed_vs_pt.jl:rho12", "Reduced rho_12 PT vs Hilbert", Dict(
                "t" => t,
                "pt_re12" => real(ρ_pt[1, 2]),
                "pt_im12" => imag(ρ_pt[1, 2]),
                "ed_re12" => real(ρ_h[1, 2]),
                "ed_im12" => imag(ρ_h[1, 2]),
            ))
        end
    end

    _agent_log("A", "debug_ed_vs_pt.jl:trace_check", "Joint vs reduced Sx over trajectory", Dict(
        "max_hilbert_liouv_or_jointred" => max_hilbert_liouv,
    ))
    _agent_log("C", "debug_ed_vs_pt.jl:split", "PT vs references max |dSx|", Dict(
        "max_pt_hilbert" => max_pt_hilbert,
        "max_pt_liouv" => max_pt_liouv,
        "max_pt_split" => max_pt_split,
        "max_sz_pt_hilbert" => max_sz_pt_hilbert,
    ))

    pt_fine = build_process_tensor(
        model.system; method=Dense(), environment=model.bath, dt=dt / 2, nsteps=2 * nsteps,
        alg=Exact(), sys_alg=Trotter{2}(), progress=false,
    )
    traj_fine = evolve(pt_fine, model.ρ_sys0; progress=false)
    ρ_coarse = one_site_density_matrix(traj.states_hilbert[end])
    ρ_fine = one_site_density_matrix(traj_fine.states_hilbert[end])
    _agent_log("E", "debug_ed_vs_pt.jl:dt", "Dense PT dt vs dt/2 at same physical time", Dict(
        "t" => nsteps * dt,
        "sx_dt" => real(tr(ρ_coarse * model.Sx)),
        "sx_dthalf" => real(tr(ρ_fine * model.Sx)),
        "absdiff" => abs(real(tr(ρ_coarse * model.Sx)) - real(tr(ρ_fine * model.Sx))),
    ))
    return nothing
end

main()
