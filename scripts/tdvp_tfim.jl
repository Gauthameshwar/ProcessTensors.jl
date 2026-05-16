# Liouville-space TDVP benchmark for the transverse-field Ising model (TFIM).
# Reference: dense Liouvillian ED, vec(ρ(t)) = exp(t L) vec(ρ(0)).
# Algorithm: 1-site TDVP, 2-site TDVP, and dynamic 2→1 TDVP.
#
# Run from repo root:
#   julia --project=. scripts/tdvp_tfim.jl

using ProcessTensors, ITensors, LinearAlgebra
using CairoMakie, LaTeXStrings

const _ROOT = @__DIR__

# -------- self-contained dense / MPO helpers -----------------------------------

function _hilbert_mpo_to_dense(ρ::AbstractMPO{Hilbert}, physical_sites)
    T = ρ.core[1]
    for j in 2:length(ρ.core)
        T *= ρ.core[j]
    end
    A = Array(T, prime.(physical_sites)..., physical_sites...)
    return reshape(ComplexF64.(A), prod(dim.(physical_sites)), prod(dim.(physical_sites)))
end

function _hilbert_matrix_to_mpo(M::AbstractMatrix{<:Number}, physical_sites)
    dims = vcat(dim.(prime.(physical_sites)), dim.(physical_sites))
    T = ITensor(reshape(ComplexF64.(M), Tuple(dims)), prime.(physical_sites)..., physical_sites...)
    return MPO(T, physical_sites)
end

function _liouville_state_to_dense(ρ_vec::AbstractMPS{Liouville}, physical_sites)
    return _hilbert_mpo_to_dense(to_hilbert(ρ_vec), physical_sites)
end

function _dense_liouvillian_matrix(os_H::OpSum, jump_ops, physical_sites, liouv_sites_shared)
    L_mpo = MPO_Liouville(os_H, liouv_sites_shared; jump_ops=jump_ops)
    d = prod(dim.(physical_sites))
    d2 = d * d
    L_dense = zeros(ComplexF64, d2, d2)
    for b in 1:d, a in 1:d
        q = a + (b - 1) * d
        E = zeros(ComplexF64, d, d)
        E[a, b] = 1.0
        basis_q = to_liouville(_hilbert_matrix_to_mpo(E, physical_sites); sites=liouv_sites_shared)
        σ_q = apply(L_mpo, basis_q; cutoff=0.0, maxdim=typemax(Int))
        L_dense[:, q] = vec(_liouville_state_to_dense(σ_q, physical_sites))
    end
    return L_dense
end

function _dense_hamiltonian_matrix(os_H::OpSum, physical_sites)
    return _hilbert_mpo_to_dense(MPO(os_H, physical_sites), physical_sites)
end

function _single_site_pauli_mpos(op::AbstractString, physical_sites)
    N = length(physical_sites)
    return MPO{Hilbert}[
        let os = OpSum()
            os += 1.0, op, j
            MPO(os, physical_sites)
        end for j in 1:N
    ]
end

function _mean_pauli_trace_mpo(ρ_vec::MPS{Liouville}, pauli_mpos::Vector{MPO{Hilbert}})
    ρ_h = to_hilbert(ρ_vec)
    s = 0.0
    for O in pauli_mpos
        ρO = apply(O, ρ_h; alg="naive", truncate=false)
        s += real(tr(ρO))
    end
    return s / length(pauli_mpos)
end

function _vectorized_identity_state(physical_sites, liouv_sites_shared)
    d = prod(dim.(physical_sites))
    identity_mpo = _hilbert_matrix_to_mpo(Matrix{ComplexF64}(I, d, d), physical_sites)
    return to_liouville(identity_mpo; sites=liouv_sites_shared)
end

function _liouville_trace(ρ_vec::AbstractMPS{Liouville}, trace_bra::AbstractMPS{Liouville})
    return inner(trace_bra, ρ_vec)
end

function _expectation_trace_mpo(ρ_vec::MPS{Liouville}, O::MPO{Hilbert})
    ρ_h = to_hilbert(ρ_vec)
    return real(tr(apply(O, ρ_h; alg="naive", truncate=false)))
end

function _energy_expectation_mpo(ρ_vec::MPS{Liouville}, H_mpo::MPO{Hilbert})
    return _expectation_trace_mpo(ρ_vec, H_mpo)
end

function _dense_density_metrics(ρ_dense::AbstractMatrix{<:Number})
    ρ = ComplexF64.(ρ_dense)
    herm_defect = norm(ρ - ρ') / max(norm(ρ), eps(Float64))
    ρ_herm = (ρ + ρ') / 2
    λmin = minimum(real.(eigvals(Hermitian(ρ_herm))))
    return (trace=tr(ρ), hermiticity=herm_defect, min_eig=λmin)
end

# -------- model ----------------------------------------------------------------

function tfim_hamiltonian(N::Int; J::Float64=1.0, h::Float64=1.2)
    os_H = OpSum()
    for j in 1:(N - 1)
        os_H += -J, "Z", j, "Z", j + 1
    end
    for j in 1:N
        os_H += -h, "X", j
    end
    return os_H
end

tfim_decay_jump_ops(N::Int; γ::Float64=0.5) = [(γ, "S-", j) for j in 1:N]

function dense_one_site_operator(op_name::AbstractString, physical_sites, site::Int)
    local_ops = Matrix{ComplexF64}[]
    for (j, s) in enumerate(physical_sites)
        if j == site
            push!(local_ops, Array(op(op_name, s), prime(s), s))
        else
            push!(local_ops, Matrix{ComplexF64}(I, dim(s), dim(s)))
        end
    end
    O = local_ops[1]
    for j in 2:length(local_ops)
        O = kron(O, local_ops[j])
    end
    return O
end

function average_observable_dense(ρ::AbstractMatrix{<:Number}, embedded_ops)
    return real(sum(tr(ρ * O) for O in embedded_ops) / length(embedded_ops))
end

# -------- exact ED: vec(ρ(t)) = exp(t L) vec(ρ(0)) -----------------------------

function exact_density_trajectory(L_dense, vec0::AbstractVector, d::Int, times::AbstractVector)
    densities = Matrix{ComplexF64}[]
    Lt = ComplexF64.(L_dense)
    v0 = ComplexF64.(vec0)
    for t in times
        vt = t == 0 ? v0 : exp(t * Lt) * v0
        push!(densities, reshape(vt, d, d))
    end
    return densities
end

function exact_metrics(density_trajectory, H_dense, x_ops, z_ops)
    energy, trace_err, herm, min_eig, sx, sz = Float64[], Float64[], Float64[], Float64[], Float64[], Float64[]
    energy0 = real(tr(first(density_trajectory) * H_dense))
    for ρ_dense in density_trajectory
        metrics = _dense_density_metrics(ρ_dense)
        push!(energy, real(tr(ρ_dense * H_dense)))
        push!(trace_err, abs(metrics.trace - 1))
        push!(herm, metrics.hermiticity)
        push!(min_eig, metrics.min_eig)
        push!(sx, average_observable_dense(ρ_dense, x_ops))
        push!(sz, average_observable_dense(ρ_dense, z_ops))
    end
    return (
        energy=energy,
        energy_drift=abs.(energy .- energy0),
        trace_err=trace_err,
        hermiticity=herm,
        min_eig=min_eig,
        sx=sx,
        sz=sz,
    )
end

# -------- TDVP algorithm -------------------------------------------------------

function _run_tdvp_with_progress(label::String, ρ0_vec, operator, dt::Float64, nsteps::Int; nsite::Int, maxdim::Int, cutoff::Float64)
    states = Vector{typeof(ρ0_vec)}(undef, nsteps + 1)
    states[1] = copy(ρ0_vec)
    current = copy(ρ0_vec)
    t_start = time()
    println("[$label] start | nsteps=$nsteps dt=$dt nsite=$nsite maxdim=$maxdim cutoff=$cutoff")
    for step in 1:nsteps
        current = tdvp(operator, dt, current; time_step=dt, nsite=nsite, maxdim=maxdim, cutoff=cutoff, outputlevel=0)
        states[step + 1] = current
        if step == 1 || step % 10 == 0 || step == nsteps
            println("[$label] step=$step/$nsteps t=$(round(step * dt; digits=3)) bond=$(maxlinkdim(current)) elapsed=$(round(time() - t_start; digits=1))s")
        end
    end

    println("[$label] done in $(round(time() - t_start; digits=1))s")
    return states
end

function _run_dynamic_tdvp_with_progress(
    label::String, ρ0_vec, operator, dt::Float64, nsteps::Int;
    maxdim::Int, cutoff::Float64, plateau_patience::Int=2, switch_maxdim::Union{Nothing, Int}=nothing,
)
    states = Vector{typeof(ρ0_vec)}(undef, nsteps + 1)
    nsites = Vector{Int}(undef, nsteps)
    bond_dims = Vector{Int}(undef, nsteps + 1)
    states[1] = copy(ρ0_vec)
    current = copy(ρ0_vec)
    bond_dims[1] = maxlinkdim(current)
    current_nsite = 2
    stagnant_steps = 0
    t_start = time()
    println("[$label] start | nsteps=$nsteps dt=$dt dynamic(2→1) maxdim=$maxdim cutoff=$cutoff")
    for step in 1:nsteps
        nsites[step] = current_nsite
        prev_bond_dim = maxlinkdim(current)
        current = tdvp(operator, dt, current; time_step=dt, nsite=current_nsite, maxdim=maxdim, cutoff=cutoff, outputlevel=0)
        states[step + 1] = current
        bond_dims[step + 1] = maxlinkdim(current)
        if current_nsite == 2
            stagnant_steps = bond_dims[step + 1] <= prev_bond_dim ? stagnant_steps + 1 : 0
            if stagnant_steps >= plateau_patience || (!isnothing(switch_maxdim) && bond_dims[step + 1] >= switch_maxdim)
                current_nsite = 1
                println("[$label] switch 2→1 at step=$step t=$(round(step * dt; digits=3)) bond_dim=$(bond_dims[step + 1])")
            end
        end
        if step == 1 || step % 10 == 0 || step == nsteps
            println("[$label] step=$step/$nsteps t=$(round(step * dt; digits=3)) nsite=$(nsites[step]) bond=$(bond_dims[step + 1]) elapsed=$(round(time() - t_start; digits=1))s")
        end
    end
    println("[$label] done in $(round(time() - t_start; digits=1))s")
    return (; states, nsites, bond_dims)
end

function tdvp_methods_trajectory(ρ0_vec, operator, dt::Float64, T::Float64; maxdim::Int, cutoff::Float64)
    nsteps = round(Int, T / dt)
    times = collect(range(0.0, step=dt, length=nsteps + 1))
    one_states = _run_tdvp_with_progress("ONE", ρ0_vec, operator, dt, nsteps; nsite=1, maxdim=maxdim, cutoff=cutoff)
    two_states = _run_tdvp_with_progress("TWO", ρ0_vec, operator, dt, nsteps; nsite=2, maxdim=maxdim, cutoff=cutoff)
    dynamic = _run_dynamic_tdvp_with_progress("DYNAMIC", ρ0_vec, operator, dt, nsteps; maxdim=maxdim, cutoff=cutoff)
    return (
        times=times,
        one=(states=one_states, bond_dims=[maxlinkdim(s) for s in one_states]),
        two=(states=two_states, bond_dims=[maxlinkdim(s) for s in two_states]),
        dynamic=dynamic,
    )
end

function tdvp_method_metrics(states, physical_sites, trace_bra, H_mpo, x_mpos, z_mpos)
    energy, trace_err, herm, min_eig, sx, sz, bond_dims = Float64[], Float64[], Float64[], Float64[], Float64[], Float64[], Int[]
    energy0 = _energy_expectation_mpo(first(states), H_mpo)
    for state in states
        ρ_dense = _liouville_state_to_dense(state, physical_sites)
        metrics = _dense_density_metrics(ρ_dense)
        push!(energy, _energy_expectation_mpo(state, H_mpo))
        push!(trace_err, abs(_liouville_trace(state, trace_bra) - 1))
        push!(herm, metrics.hermiticity)
        push!(min_eig, metrics.min_eig)
        push!(sx, _mean_pauli_trace_mpo(state, x_mpos))
        push!(sz, _mean_pauli_trace_mpo(state, z_mpos))
        push!(bond_dims, maxlinkdim(state))
    end
    return (energy=energy, energy_drift=abs.(energy .- energy0), trace_err=trace_err, hermiticity=herm, min_eig=min_eig, sx=sx, sz=sz, bond_dims=bond_dims)
end

const _METHOD_STYLES = Dict(
    :one => (; color=:dodgerblue3, linestyle=:dash, label="1TDVP"),
    :two => (; color=:darkorange2, linestyle=:solid, label="2TDVP"),
    :dynamic => (; color=:seagreen4, linestyle=:dot, label="Dynamic TDVP"),
)

function add_method_lines!(ax, times, method_data, field::Symbol)
    handles, labels = AbstractPlot[], String[]
    for method in (:one, :two, :dynamic)
        style = _METHOD_STYLES[method]
        h = lines!(ax, times, getfield(method_data[method], field); color=style.color, linestyle=style.linestyle, linewidth=2.6)
        push!(handles, h)
        push!(labels, style.label)
    end
    return handles, labels
end

function plot_observable(path, times, exact_times, exact_curve, method_data, field::Symbol; ylabel, title)
    fig = Figure(size=(900, 520))
    ax = Axis(fig[1, 1]; xlabel=L"$t$", ylabel=ylabel, title=title)
    handles, labels = AbstractPlot[], String[]
    h_exact = lines!(ax, exact_times, exact_curve; color=:black, linewidth=4.8)
    push!(handles, h_exact)
    push!(labels, "ED")
    mh, ml = add_method_lines!(ax, times, method_data, field)
    append!(handles, mh)
    append!(labels, ml)
    Legend(fig[2, 1], handles, labels; orientation=:horizontal, nbanks=1, tellwidth=false, tellheight=true)
    save(path, fig)
    println("Saved: ", path)
end

function plot_unitary_conserved(path, times, exact_times, exact_data, method_data; title)
    fig = Figure(size=(1200, 700))
    ax_energy = Axis(fig[1, 1]; xlabel=L"$t$", ylabel=L"$\langle H \rangle (t)$", title=title)
    ax_drift = Axis(fig[1, 2]; xlabel=L"$t$", ylabel="energy drift")
    ax_trace = Axis(fig[1, 3]; xlabel=L"$t$", ylabel="trace error", yscale=log10)
    ax_herm = Axis(fig[2, 1]; xlabel=L"$t$", ylabel="Hermiticity defect", yscale=log10)
    ax_psd = Axis(fig[2, 2]; xlabel=L"$t$", ylabel="min eig")
    ax_bond = Axis(fig[2, 3]; xlabel=L"$t$", ylabel="max bond dim")

    exact_energy = lines!(ax_energy, exact_times, exact_data.energy; color=:black, linewidth=4.8)
    lines!(ax_drift, exact_times, exact_data.energy_drift; color=:black, linewidth=4.8)
    lines!(ax_trace, exact_times, max.(exact_data.trace_err, eps(Float64)); color=:black, linewidth=4.8)
    lines!(ax_herm, exact_times, max.(exact_data.hermiticity, eps(Float64)); color=:black, linewidth=4.8)
    lines!(ax_psd, exact_times, exact_data.min_eig; color=:black, linewidth=4.8)

    handles, labels = AbstractPlot[exact_energy], ["ED"]
    for method in (:one, :two, :dynamic)
        style = _METHOD_STYLES[method]
        data = method_data[method]
        h = lines!(ax_energy, times, data.energy; color=style.color, linestyle=style.linestyle, linewidth=2.4)
        lines!(ax_drift, times, max.(data.energy_drift, eps(Float64)); color=style.color, linestyle=style.linestyle, linewidth=2.4)
        lines!(ax_trace, times, max.(data.trace_err, eps(Float64)); color=style.color, linestyle=style.linestyle, linewidth=2.4)
        lines!(ax_herm, times, max.(data.hermiticity, eps(Float64)); color=style.color, linestyle=style.linestyle, linewidth=2.4)
        lines!(ax_psd, times, data.min_eig; color=style.color, linestyle=style.linestyle, linewidth=2.4)
        lines!(ax_bond, times, data.bond_dims; color=style.color, linestyle=style.linestyle, linewidth=2.4)
        push!(handles, h)
        push!(labels, style.label)
    end
    Legend(fig[3, 1:3], handles, labels; orientation=:horizontal, nbanks=1, tellwidth=false, tellheight=true)
    save(path, fig)
    println("Saved: ", path)
end

function plot_dissipative_conserved(path, times, exact_times, exact_data, method_data; title)
    fig = Figure(size=(1100, 700))
    ax_trace = Axis(fig[1, 1]; xlabel=L"$t$", ylabel="trace error", title=title, yscale=log10)
    ax_herm = Axis(fig[1, 2]; xlabel=L"$t$", ylabel="Hermiticity defect", yscale=log10)
    ax_psd = Axis(fig[2, 1]; xlabel=L"$t$", ylabel="min eig")
    ax_bond = Axis(fig[2, 2]; xlabel=L"$t$", ylabel="max bond dim")

    exact_trace = lines!(ax_trace, exact_times, max.(exact_data.trace_err, eps(Float64)); color=:black, linewidth=4.8)
    lines!(ax_herm, exact_times, max.(exact_data.hermiticity, eps(Float64)); color=:black, linewidth=4.8)
    lines!(ax_psd, exact_times, exact_data.min_eig; color=:black, linewidth=4.8)

    handles, labels = AbstractPlot[exact_trace], ["ED"]
    for method in (:one, :two, :dynamic)
        style = _METHOD_STYLES[method]
        data = method_data[method]
        h = lines!(ax_trace, times, max.(data.trace_err, eps(Float64)); color=style.color, linestyle=style.linestyle, linewidth=2.4)
        lines!(ax_herm, times, max.(data.hermiticity, eps(Float64)); color=style.color, linestyle=style.linestyle, linewidth=2.4)
        lines!(ax_psd, times, data.min_eig; color=style.color, linestyle=style.linestyle, linewidth=2.4)
        lines!(ax_bond, times, data.bond_dims; color=style.color, linestyle=style.linestyle, linewidth=2.4)
        push!(handles, h)
        push!(labels, style.label)
    end
    Legend(fig[3, 1:2], handles, labels; orientation=:horizontal, nbanks=1, tellwidth=false, tellheight=true)
    save(path, fig)
    println("Saved: ", path)
end

function main()
    println("=== Liouville-space TDVP benchmark: TFIM unitary + dissipative ===")

    ##############  SETUP  ##############
    N = 4
    T_unitary, T_diss = 4.0, 9.0
    dt = 0.05
    maxdim, cutoff = 128, 1e-10

    physical_sites = siteinds("S=1/2", N)
    liouv_sites_shared = liouv_sites(physical_sites)
    trace_bra = _vectorized_identity_state(physical_sites, liouv_sites_shared)
    os_H = tfim_hamiltonian(N; J=1.0, h=1.2)
    H_mpo = MPO(os_H, physical_sites)
    H_dense = _dense_hamiltonian_matrix(os_H, physical_sites)
    jump_ops = tfim_decay_jump_ops(N; γ=0.5)

    ψ0 = MPS(physical_sites, fill("Up", N))
    ρ0 = to_dm(ψ0)
    ρ0_vec = to_liouville(ρ0; sites=liouv_sites_shared)
    d = prod(dim.(physical_sites))
    vec0 = vec(ComplexF64.(_hilbert_mpo_to_dense(ρ0, physical_sites)))

    x_ops = [dense_one_site_operator("X", physical_sites, j) for j in 1:N]
    z_ops = [dense_one_site_operator("Z", physical_sites, j) for j in 1:N]
    x_mpos = _single_site_pauli_mpos("X", physical_sites)
    z_mpos = _single_site_pauli_mpos("Z", physical_sites)

    L_unitary_mpo = MPO_Liouville(os_H, liouv_sites_shared; jump_ops=Tuple{Number, String, Int}[])
    L_diss_mpo = MPO_Liouville(os_H, liouv_sites_shared; jump_ops=jump_ops)
    L_unitary_dense = _dense_liouvillian_matrix(os_H, Tuple{Number, String, Int}[], physical_sites, liouv_sites_shared)
    L_diss_dense = _dense_liouvillian_matrix(os_H, jump_ops, physical_sites, liouv_sites_shared)

    ##############  TDVP  ##############
    unitary_runs = tdvp_methods_trajectory(ρ0_vec, L_unitary_mpo, dt, T_unitary; maxdim=maxdim, cutoff=cutoff)
    dissipative_runs = tdvp_methods_trajectory(ρ0_vec, L_diss_mpo, dt, T_diss; maxdim=maxdim, cutoff=cutoff)

    ##############  EXACT ED (exp(tL))  ##############
    unitary_exact_densities = exact_density_trajectory(L_unitary_dense, vec0, d, unitary_runs.times)
    dissipative_exact_densities = exact_density_trajectory(L_diss_dense, vec0, d, dissipative_runs.times)
    unitary_exact = exact_metrics(unitary_exact_densities, H_dense, x_ops, z_ops)
    dissipative_exact = exact_metrics(dissipative_exact_densities, H_dense, x_ops, z_ops)

    ##############  TDVP METRICS  ##############
    unitary_method_data = Dict(
        :one => tdvp_method_metrics(unitary_runs.one.states, physical_sites, trace_bra, H_mpo, x_mpos, z_mpos),
        :two => tdvp_method_metrics(unitary_runs.two.states, physical_sites, trace_bra, H_mpo, x_mpos, z_mpos),
        :dynamic => tdvp_method_metrics(unitary_runs.dynamic.states, physical_sites, trace_bra, H_mpo, x_mpos, z_mpos),
    )
    dissipative_method_data = Dict(
        :one => tdvp_method_metrics(dissipative_runs.one.states, physical_sites, trace_bra, H_mpo, x_mpos, z_mpos),
        :two => tdvp_method_metrics(dissipative_runs.two.states, physical_sites, trace_bra, H_mpo, x_mpos, z_mpos),
        :dynamic => tdvp_method_metrics(dissipative_runs.dynamic.states, physical_sites, trace_bra, H_mpo, x_mpos, z_mpos),
    )

    ##############  PLOTTING  ##############
    figdir = joinpath(_ROOT, "figures")
    mkpath(figdir)

    plot_observable(
        joinpath(figdir, "tdvp_tfim_unitary_mx.png"), unitary_runs.times, unitary_runs.times, unitary_exact.sx,
        unitary_method_data, :sx; ylabel=L"$\langle \overline{\sigma}_x \rangle (t)$", title="Unitary TFIM TDVP (N=$N, dt=$dt)",
    )
    plot_observable(
        joinpath(figdir, "tdvp_tfim_unitary_mz.png"), unitary_runs.times, unitary_runs.times, unitary_exact.sz,
        unitary_method_data, :sz; ylabel=L"$\langle \overline{\sigma}_z \rangle (t)$", title="Unitary TFIM TDVP (N=$N, dt=$dt)",
    )
    plot_unitary_conserved(
        joinpath(figdir, "tdvp_tfim_unitary_conserved.png"), unitary_runs.times, unitary_runs.times, unitary_exact,
        unitary_method_data; title="Unitary TFIM conserved quantities",
    )
    plot_observable(
        joinpath(figdir, "tdvp_tfim_dissipative_mx.png"), dissipative_runs.times, dissipative_runs.times, dissipative_exact.sx,
        dissipative_method_data, :sx; ylabel=L"$\langle \overline{\sigma}_x \rangle (t)$", title="Dissipative TFIM TDVP (N=$N, dt=$dt)",
    )
    plot_observable(
        joinpath(figdir, "tdvp_tfim_dissipative_mz.png"), dissipative_runs.times, dissipative_runs.times, dissipative_exact.sz,
        dissipative_method_data, :sz; ylabel=L"$\langle \overline{\sigma}_z \rangle (t)$", title="Dissipative TFIM TDVP (N=$N, dt=$dt)",
    )
    plot_dissipative_conserved(
        joinpath(figdir, "tdvp_tfim_dissipative_conserved.png"), dissipative_runs.times, dissipative_runs.times,
        dissipative_exact, dissipative_method_data; title="Dissipative TFIM physicality diagnostics",
    )
    
    println("✓ Done")
end

main()
