# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: scripts/tdvp_tfim_unitary.jl
# Contributor: Gauthameshwar S.
#
# Unitary TFIM (N=4): Hilbert- and Liouville-space TDVP with plain 1-TDVP,
# 1-TDVP + global subspace expansion, and 2-TDVP against dense Schrödinger evolution.
#
# Run with:
#   julia --project=. scripts/tdvp_tfim_unitary.jl

using Printf
using ProcessTensors
using ITensors
import ITensorMPS
using LinearAlgebra
using Statistics: mean
using CairoMakie
using LaTeXStrings

# ------------------------------------------------------------------------------
# 1. Small script utilities
# ------------------------------------------------------------------------------

const STATUS_WIDTH = 90

function print_section(title::AbstractString)
    println()
    println(title)
    println("-" ^ length(title))
end

function update_status(message::AbstractString)
    print("\r", rpad(message, STATUS_WIDTH))
    flush(stdout)
end

function finish_status(message::AbstractString = "")
    print("\r", " "^STATUS_WIDTH, "\r")
    if !isempty(message)
        println(message)
    end
    flush(stdout)
end

function hilbert_mpo_to_dense(ρ::AbstractMPO{Hilbert}, physical_sites)
    T = foldl(*, ρ)
    A = Array(T, prime.(physical_sites)..., physical_sites...)
    return reshape(ComplexF64.(A), prod(dim.(physical_sites)), prod(dim.(physical_sites)))
end

function hilbert_matrix_to_mpo(M::AbstractMatrix{<:Number}, physical_sites)
    dims = vcat(dim.(prime.(physical_sites)), dim.(physical_sites))
    T = ITensor(reshape(ComplexF64.(M), Tuple(dims)), prime.(physical_sites)..., physical_sites...)
    return MPO(T, physical_sites)
end

function hilbert_mps_to_dense(ψ::AbstractMPS{Hilbert}, physical_sites)
    T = foldl(*, ψ)
    return vec(ComplexF64.(Array(T, physical_sites...)))
end

function liouville_state_to_dense(ρ_vec::AbstractMPS{Liouville}, physical_sites)
    return hilbert_mpo_to_dense(to_hilbert(ρ_vec), physical_sites)
end

function dense_liouvillian_matrix(os_H::OpSum, jump_ops, physical_sites, liouv_sites_shared)
    L_mpo = liouvillian_mpo(os_H, liouv_sites_shared; jump_ops=jump_ops)
    d = prod(dim.(physical_sites))
    d2 = d * d
    L_dense = zeros(ComplexF64, d2, d2)
    for b in 1:d, a in 1:d
        q = a + (b - 1) * d
        E = zeros(ComplexF64, d, d)
        E[a, b] = 1.0
        basis_q = to_liouville(hilbert_matrix_to_mpo(E, physical_sites); sites=liouv_sites_shared)
        σ_q = apply(L_mpo, basis_q; cutoff=0.0, maxdim=typemax(Int))
        L_dense[:, q] = vec(liouville_state_to_dense(σ_q, physical_sites))
    end
    return L_dense
end

function dense_hamiltonian_matrix(os_H::OpSum, physical_sites)
    return hilbert_mpo_to_dense(MPO(os_H, physical_sites), physical_sites)
end

function single_site_pauli_mpos(op::AbstractString, physical_sites)
    N = length(physical_sites)
    return MPO{Hilbert}[
        let os = OpSum()
            os += 1.0, op, j
            MPO(os, physical_sites)
        end for j in 1:N
    ]
end

function mean_pauli_trace_mpo(ρ_vec::MPS{Liouville}, pauli_mpos::Vector{MPO{Hilbert}})
    ρ_h = to_hilbert(ρ_vec)
    s = 0.0
    for O in pauli_mpos
        ρO = apply(O, ρ_h; alg="naive", truncate=false)
        s += real(tr(ρO))
    end
    return s / length(pauli_mpos)
end

function vectorized_identity_state(physical_sites, liouv_sites_shared)
    d = prod(dim.(physical_sites))
    identity_mpo = hilbert_matrix_to_mpo(Matrix{ComplexF64}(I, d, d), physical_sites)
    return to_liouville(identity_mpo; sites=liouv_sites_shared)
end

function liouville_trace(ρ_vec::AbstractMPS{Liouville}, trace_bra::AbstractMPS{Liouville})
    return inner(trace_bra, ρ_vec)
end

function energy_expectation_mpo(ρ_vec::MPS{Liouville}, H_mpo::MPO{Hilbert})
    ρ_h = to_hilbert(ρ_vec)
    return real(tr(apply(H_mpo, ρ_h; alg="naive", truncate=false)))
end

function dense_density_metrics(ρ_dense::AbstractMatrix{<:Number})
    ρ = ComplexF64.(ρ_dense)
    herm_defect = norm(ρ - ρ') / max(norm(ρ), eps(Float64))
    ρ_herm = (ρ + ρ') / 2
    λmin = minimum(real.(eigvals(Hermitian(ρ_herm))))
    return (trace=tr(ρ), hermiticity=herm_defect, min_eig=λmin)
end

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

function dense_one_site_operator(op_name::AbstractString, physical_sites, site::Int)
    local_ops = Matrix{ComplexF64}[]
    for (j, s) in enumerate(physical_sites)
        if j == site
            push!(local_ops, Array(op(op_name, s), prime(s), s))
        else
            push!(local_ops, Matrix{ComplexF64}(I, dim(s), dim(s)))
        end
    end
    return foldl(kron, local_ops)
end

function average_observable_dense(ρ::AbstractMatrix{<:Number}, embedded_ops)
    return real(sum(tr(ρ * O) for O in embedded_ops) / length(embedded_ops))
end

density_error(ρ::AbstractMatrix, ρ_ref::AbstractMatrix) =
    norm(ComplexF64.(ρ) - ComplexF64.(ρ_ref)) / max(norm(ComplexF64.(ρ_ref)), eps(Float64))

function exact_density_trajectory(H_dense, ψ0_dense::AbstractVector, times::AbstractVector)
    densities = Matrix{ComplexF64}[]
    Hd = ComplexF64.(H_dense)
    ψ0 = ComplexF64.(ψ0_dense)
    for t in times
        ψt = t == 0 ? ψ0 : exp(-1im * t * Hd) * ψ0
        push!(densities, ψt * ψt')
    end
    return densities
end

function exact_metrics(density_trajectory, H_dense, x_ops, z_ops)
    energy, trace_err, herm, min_eig, sx, sz = Float64[], Float64[], Float64[], Float64[], Float64[], Float64[]
    energy0 = real(tr(first(density_trajectory) * H_dense))
    for ρ_dense in density_trajectory
        metrics = dense_density_metrics(ρ_dense)
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

state_density_dense(state::AbstractMPS{Hilbert}, physical_sites) = let ψ = hilbert_mps_to_dense(state, physical_sites)
    ψ * ψ'
end
state_density_dense(state::AbstractMPS{Liouville}, physical_sites) = liouville_state_to_dense(state, physical_sites)

state_energy(state::AbstractMPS{Hilbert}, H_mpo) = real(inner(state', H_mpo, state))
state_energy(state::AbstractMPS{Liouville}, H_mpo) = energy_expectation_mpo(state, H_mpo)

state_mean_pauli(state::AbstractMPS{Hilbert}, pauli_mpos) = mean(real(inner(state', O, state)) for O in pauli_mpos)
state_mean_pauli(state::AbstractMPS{Liouville}, pauli_mpos) = mean_pauli_trace_mpo(state, pauli_mpos)

function gse_expand_state(
    state::MPS{Hilbert},
    operator;
    krylovdim::Int,
    gse_cutoff::Float64,
    gse_maxdim::Int,
)
    expanded_core = ITensorMPS.expand(
        state.core,
        operator.core;
        alg="global_krylov",
        krylovdim=krylovdim,
        cutoff=gse_cutoff,
        apply_kwargs=(; maxdim=gse_maxdim),
    )
    ITensorMPS.orthogonalize!(expanded_core, 1)
    return MPS{Hilbert}(expanded_core)
end

function gse_expand_state(
    state::MPS{Liouville},
    operator;
    krylovdim::Int,
    gse_cutoff::Float64,
    gse_maxdim::Int,
)
    expanded_core = ITensorMPS.expand(
        state.core,
        operator.core;
        alg="global_krylov",
        krylovdim=krylovdim,
        cutoff=gse_cutoff,
        apply_kwargs=(; maxdim=gse_maxdim),
    )
    ITensorMPS.orthogonalize!(expanded_core, 1)
    return MPS{Liouville}(expanded_core, state.combiners)
end

function tdvp_trajectory(state0, operator, time_step, dt::Float64, nsteps::Int; nsite::Int, maxdim::Int, cutoff::Float64, label::AbstractString)
    states = Vector{typeof(state0)}(undef, nsteps + 1)
    states[1] = copy(state0)
    current = copy(state0)
    progress_stride = max(1, nsteps ÷ 20)
    for step in 1:nsteps
        current = tdvp(operator, time_step, current; time_step=time_step, nsite=nsite, maxdim=maxdim, cutoff=cutoff, outputlevel=0)
        states[step + 1] = current
        if step == 1 || step == nsteps || step % progress_stride == 0
            update_status(@sprintf("  %-18s step %4d / %4d  t = %.3f  bond = %d", label, step, nsteps, step * dt, maxlinkdim(current)))
        end
    end
    finish_status(@sprintf("  %-18s complete (%d snapshots)", label, length(states)))
    return states
end

function tdvp_run_metrics(states, physical_sites, H_mpo, x_mpos, z_mpos, ρ_ed_trajectory)
    energy, trace_err, herm, min_eig, sx, sz, bond_dims, rho_errs =
        Float64[], Float64[], Float64[], Float64[], Float64[], Float64[], Int[], Float64[]
    energy0 = state_energy(first(states), H_mpo)
    for (ρ_ed, state) in zip(ρ_ed_trajectory, states)
        ρ_dense = state_density_dense(state, physical_sites)
        metrics = dense_density_metrics(ρ_dense)
        push!(energy, state_energy(state, H_mpo))
        push!(trace_err, abs(metrics.trace - 1))
        push!(herm, metrics.hermiticity)
        push!(min_eig, metrics.min_eig)
        push!(sx, state_mean_pauli(state, x_mpos))
        push!(sz, state_mean_pauli(state, z_mpos))
        push!(bond_dims, maxlinkdim(state))
        push!(rho_errs, density_error(ρ_dense, ρ_ed))
    end
    return (
        energy=energy,
        energy_drift=abs.(energy .- energy0),
        trace_err=trace_err,
        hermiticity=herm,
        min_eig=min_eig,
        sx=sx,
        sz=sz,
        bond_dims=bond_dims,
        rho_errs=rho_errs,
    )
end

function tdvp1_gse_trajectory(
    state0,
    operator,
    time_step,
    dt::Float64,
    nsteps::Int;
    maxdim::Int,
    cutoff::Float64,
    krylovdim::Int,
    gse_cutoff::Float64,
    gse_maxdim::Int,
    gse_every_steps::Int,
    label::AbstractString,
)
    states = Vector{typeof(state0)}(undef, nsteps + 1)
    states[1] = copy(state0)
    current = copy(state0)
    progress_stride = max(1, nsteps ÷ 20)
    for step in 1:nsteps
        if step == 1 || (gse_every_steps > 0 && (step - 1) % gse_every_steps == 0)
            current = gse_expand_state(
                current,
                operator;
                krylovdim=krylovdim,
                gse_cutoff=gse_cutoff,
                gse_maxdim=gse_maxdim,
            )
        end
        current = tdvp(
            operator,
            time_step,
            current;
            time_step=time_step,
            nsite=1,
            maxdim=maxdim,
            cutoff=cutoff,
            outputlevel=0,
        )
        states[step + 1] = current
        if step == 1 || step == nsteps || step % progress_stride == 0
            update_status(@sprintf("  %-24s step %4d / %4d  t = %.3f  bond = %d", label, step, nsteps, step * dt, maxlinkdim(current)))
        end
    end
    finish_status(@sprintf("  %-24s complete (%d snapshots)", label, length(states)))
    return states
end

function max_curve_error(exact::AbstractVector, approx::AbstractVector)
    return maximum(abs.(exact .- approx); init=0.0)
end

function print_series_diagnostics(label::AbstractString, exact, metrics)
    err_x = max_curve_error(exact.sx, metrics.sx)
    err_z = max_curve_error(exact.sz, metrics.sz)
    max_drift = maximum(metrics.energy_drift; init=0.0)
    println(label)
    @printf("    max |⟨σ_x⟩−ED| = %.3e\n", err_x)
    @printf("    max |⟨σ_z⟩−ED| = %.3e\n", err_z)
    @printf("    max energy drift = %.3e\n", max_drift)
    return (err_x, err_z, max_drift)
end

const METHOD_COLORS = Dict(
    :plain => :dodgerblue3,
    :gse10 => :seagreen3,
    :tdvp2 => :darkorange2,
)
const ED_LINEWIDTH = 3.6
const TDVP2_LINEWIDTH = 2.0
const TDVP1_LINEWIDTH = 2.4

function plot_observable_comparison(path, times, exact_curve, run_series, observable::Symbol; ylabel, title, space=nothing)
    fig = Figure(size=(900, 520))
    ax = Axis(fig[1, 1]; xlabel=L"$t$", ylabel=ylabel, title=title)
    handles, labels = AbstractPlot[], Any[]
    h_ed = lines!(ax, times, exact_curve; color=:black, linewidth=ED_LINEWIDTH)
    push!(handles, h_ed)
    push!(labels, L"$\mathrm{ED}\,(e^{t L})$")
    series_list = space === nothing ? run_series : filter(series -> series.space == space, run_series)
    for series in series_list
        h = lines!(
            ax,
            times,
            getfield(series.metrics, observable);
            color=METHOD_COLORS[series.method],
            linestyle=series.space == :liouville ? :solid : :dot,
            linewidth=series.method == :tdvp2 ? TDVP2_LINEWIDTH : TDVP1_LINEWIDTH,
        )
        push!(handles, h)
        push!(labels, series.label)
    end
    axislegend(ax, handles, labels; position=:rt, nbanks=2, fontsize=10)
    save(path, fig)
    return path
end

function plot_rho_error_comparison(path, times, run_series; title, space)
    fig = Figure(size=(900, 520))
    ax = Axis(
        fig[1, 1];
        xlabel=L"$t$",
        ylabel="||ρ - ρ_ED|| / ||ρ_ED||",
        title=title,
        yscale=log10,
    )
    handles, labels = AbstractPlot[], Any[]
    for series in filter(series -> series.space == space, run_series)
        h = lines!(
            ax,
            times,
            max.(series.metrics.rho_errs, eps(Float64));
            color=METHOD_COLORS[series.method],
            linestyle=series.space == :liouville ? :solid : :dot,
            linewidth=series.method == :tdvp2 ? TDVP2_LINEWIDTH : TDVP1_LINEWIDTH,
        )
        push!(handles, h)
        push!(labels, series.label)
    end
    axislegend(ax, handles, labels; position=:rt, nbanks=2, fontsize=10)
    save(path, fig)
    return path
end

function plot_energy_drift_comparison(path, times, run_series; title, space)
    fig = Figure(size=(900, 520))
    ax = Axis(
        fig[1, 1];
        xlabel=L"$t$",
        ylabel=L"$\langle H \rangle(t) - \langle H \rangle(0)$",
        title=title,
    )
    hlines!(ax, [0.0]; color=:black, linewidth=1.2, linestyle=:dash)
    handles, labels = AbstractPlot[], Any[]
    for series in filter(series -> series.space == space, run_series)
        drift = series.metrics.energy .- series.metrics.energy[1]
        h = lines!(
            ax,
            times,
            drift;
            color=METHOD_COLORS[series.method],
            linestyle=series.space == :liouville ? :solid : :dot,
            linewidth=series.method == :tdvp2 ? TDVP2_LINEWIDTH : TDVP1_LINEWIDTH,
        )
        push!(handles, h)
        push!(labels, series.label)
    end
    axislegend(ax, handles, labels; position=:rt, nbanks=2, fontsize=10)
    save(path, fig)
    return path
end

function plot_unitary_conserved(path, times, exact_data, run_series; title)
    fig = Figure(size=(1200, 700))
    ax_energy = Axis(fig[1, 1]; xlabel=L"$t$", ylabel=L"$\langle H \rangle (t)$", title=title)
    ax_drift = Axis(fig[1, 2]; xlabel=L"$t$", ylabel="energy drift")
    ax_trace = Axis(fig[1, 3]; xlabel=L"$t$", ylabel="trace error", yscale=log10)
    ax_herm = Axis(fig[2, 1]; xlabel=L"$t$", ylabel="Hermiticity defect", yscale=log10)
    ax_psd = Axis(fig[2, 2]; xlabel=L"$t$", ylabel="min eig")
    ax_bond = Axis(fig[2, 3]; xlabel=L"$t$", ylabel="max bond dim")

    exact_energy = lines!(ax_energy, times, exact_data.energy; color=:black, linewidth=ED_LINEWIDTH)
    lines!(ax_drift, times, exact_data.energy_drift; color=:black, linewidth=ED_LINEWIDTH)
    lines!(ax_trace, times, max.(exact_data.trace_err, eps(Float64)); color=:black, linewidth=ED_LINEWIDTH)
    lines!(ax_herm, times, max.(exact_data.hermiticity, eps(Float64)); color=:black, linewidth=ED_LINEWIDTH)
    lines!(ax_psd, times, exact_data.min_eig; color=:black, linewidth=ED_LINEWIDTH)

    handles, labels = AbstractPlot[exact_energy], [L"$\mathrm{ED}$"]
    for series in filter(series -> series.space == :liouville, run_series)
        data = series.metrics
        linewidth = series.method == :tdvp2 ? TDVP2_LINEWIDTH : TDVP1_LINEWIDTH
        h = lines!(ax_energy, times, data.energy; color=METHOD_COLORS[series.method], linestyle=:solid, linewidth=linewidth)
        lines!(ax_drift, times, max.(data.energy_drift, eps(Float64)); color=METHOD_COLORS[series.method], linestyle=:solid, linewidth=linewidth)
        lines!(ax_trace, times, max.(data.trace_err, eps(Float64)); color=METHOD_COLORS[series.method], linestyle=:solid, linewidth=linewidth)
        lines!(ax_herm, times, max.(data.hermiticity, eps(Float64)); color=METHOD_COLORS[series.method], linestyle=:solid, linewidth=linewidth)
        lines!(ax_psd, times, data.min_eig; color=METHOD_COLORS[series.method], linestyle=:solid, linewidth=linewidth)
        lines!(ax_bond, times, data.bond_dims; color=METHOD_COLORS[series.method], linestyle=:solid, linewidth=linewidth)
        push!(handles, h)
        push!(labels, series.label)
    end
    for series in filter(series -> series.space == :hilbert, run_series)
        data = series.metrics
        linewidth = series.method == :tdvp2 ? TDVP2_LINEWIDTH : TDVP1_LINEWIDTH
        h = lines!(ax_energy, times, data.energy; color=METHOD_COLORS[series.method], linestyle=:dot, linewidth=linewidth)
        lines!(ax_drift, times, max.(data.energy_drift, eps(Float64)); color=METHOD_COLORS[series.method], linestyle=:dot, linewidth=linewidth)
        lines!(ax_trace, times, max.(data.trace_err, eps(Float64)); color=METHOD_COLORS[series.method], linestyle=:dot, linewidth=linewidth)
        lines!(ax_herm, times, max.(data.hermiticity, eps(Float64)); color=METHOD_COLORS[series.method], linestyle=:dot, linewidth=linewidth)
        lines!(ax_psd, times, data.min_eig; color=METHOD_COLORS[series.method], linestyle=:dot, linewidth=linewidth)
        lines!(ax_bond, times, data.bond_dims; color=METHOD_COLORS[series.method], linestyle=:dot, linewidth=linewidth)
        push!(handles, h)
        push!(labels, series.label)
    end
    Legend(fig[3, 1:3], handles, labels; orientation=:horizontal, nbanks=2, tellwidth=false, tellheight=true)
    save(path, fig)
    return path
end

# ------------------------------------------------------------------------------
# 2. Script parameters
# ------------------------------------------------------------------------------

const N = 4
const J = 1.0
const h = 1.2
const T_max = 4.0
const dt = 0.05
const nsteps = round(Int, T_max / dt)
const maxdim_1site = 50
const maxdim_2site = 50
const gse_every_steps = 10
const krylovdim = 2
const gse_cutoff = 1e-8
const cutoff = 1e-10

output_dir = joinpath(@__DIR__, "figures")
mkpath(output_dir)

# ------------------------------------------------------------------------------
# 3. Define the physical problem
# ------------------------------------------------------------------------------

print_section("Problem setup")

println("Unitary transverse-field Ising chain: Hilbert- and Liouville-space TDVP vs dense Schrödinger evolution.")
println()

@printf("chain length N        = %d\n", N)
@printf("J, h                  = %.3f, %.3f\n", J, h)
@printf("final time T_max      = %.3f\n", T_max)
@printf("time step dt          = %.3f\n", dt)
@printf("1-TDVP max bond dim   = %d\n", maxdim_1site)
@printf("2-TDVP max bond dim   = %d\n", maxdim_2site)
@printf("GSE every-n steps     = %d\n", gse_every_steps)
@printf("GSE Krylov dim        = %d\n", krylovdim)
@printf("GSE cutoff            = %.1e\n", gse_cutoff)
@printf("SVD cutoff            = %.1e\n", cutoff)

physical_sites = siteinds("S=1/2", N)
liouv_sites_shared = liouv_sites(physical_sites)
os_H = tfim_hamiltonian(N; J=J, h=h)
H_mpo = MPO(os_H, physical_sites)
H_dense = dense_hamiltonian_matrix(os_H, physical_sites)

ψ0_hilbert = MPS(physical_sites, fill("Up", N))
ρ0 = to_dm(ψ0_hilbert)
ρ0_liouville = to_liouville(ρ0; sites=liouv_sites_shared)
ψ0_dense = hilbert_mps_to_dense(ψ0_hilbert, physical_sites)

x_ops = [dense_one_site_operator("X", physical_sites, j) for j in 1:N]
z_ops = [dense_one_site_operator("Z", physical_sites, j) for j in 1:N]
x_mpos = single_site_pauli_mpos("X", physical_sites)
z_mpos = single_site_pauli_mpos("Z", physical_sites)
times = collect(range(0.0, step=dt, length=nsteps + 1))

# ------------------------------------------------------------------------------
# 4. Main computation
# ------------------------------------------------------------------------------

print_section("Main computation")

println("Building dense Hilbert-space reference...")
exact_densities = exact_density_trajectory(H_dense, ψ0_dense, times)
exact = exact_metrics(exact_densities, H_dense, x_ops, z_ops)

println("Running Hilbert- and Liouville-space TDVP sweeps...")
run_series = NamedTuple{(:label, :method, :space, :metrics), Tuple{String, Symbol, Symbol, Any}}[]

label_plain_h = "1-TDVP plain Hilbert"
states_plain_h = tdvp_trajectory(
    ψ0_hilbert,
    H_mpo,
    -1im * dt,
    dt,
    nsteps;
    nsite=1,
    maxdim=maxdim_1site,
    cutoff=cutoff,
    label=label_plain_h,
)
metrics_plain_h = tdvp_run_metrics(states_plain_h, physical_sites, H_mpo, x_mpos, z_mpos, exact_densities)
push!(run_series, (; label=label_plain_h, method=:plain, space=:hilbert, metrics=metrics_plain_h))

label_gse_h = "1-TDVP+GSE Hilbert"
states_gse_h = tdvp1_gse_trajectory(
    ψ0_hilbert,
    H_mpo,
    -1im * dt,
    dt,
    nsteps;
    maxdim=maxdim_1site,
    cutoff=cutoff,
    krylovdim=krylovdim,
    gse_cutoff=gse_cutoff,
    gse_maxdim=maxdim_1site,
    label=label_gse_h,
    gse_every_steps=gse_every_steps,
)
metrics_gse_h = tdvp_run_metrics(states_gse_h, physical_sites, H_mpo, x_mpos, z_mpos, exact_densities)
push!(run_series, (; label=label_gse_h, method=:gse10, space=:hilbert, metrics=metrics_gse_h))

label_2site_h = "2-TDVP Hilbert"
states_2site_h = tdvp_trajectory(
    ψ0_hilbert,
    H_mpo,
    -1im * dt,
    dt,
    nsteps;
    nsite=2,
    maxdim=maxdim_2site,
    cutoff=cutoff,
    label=label_2site_h,
)
metrics_2site_h = tdvp_run_metrics(states_2site_h, physical_sites, H_mpo, x_mpos, z_mpos, exact_densities)
push!(run_series, (; label=label_2site_h, method=:tdvp2, space=:hilbert, metrics=metrics_2site_h))

L_mpo = liouvillian_mpo(os_H, liouv_sites_shared; jump_ops=Tuple{Number, String, Int}[])

label_plain_l = "1-TDVP plain Liouville"
states_plain_l = tdvp_trajectory(
    ρ0_liouville,
    L_mpo,
    dt,
    dt,
    nsteps;
    nsite=1,
    maxdim=maxdim_1site,
    cutoff=cutoff,
    label=label_plain_l,
)
metrics_plain_l = tdvp_run_metrics(states_plain_l, physical_sites, H_mpo, x_mpos, z_mpos, exact_densities)
push!(run_series, (; label=label_plain_l, method=:plain, space=:liouville, metrics=metrics_plain_l))

label_gse_l = "1-TDVP+GSE Liouville"
states_gse_l = tdvp1_gse_trajectory(
    ρ0_liouville,
    L_mpo,
    dt,
    dt,
    nsteps;
    maxdim=maxdim_1site,
    cutoff=cutoff,
    krylovdim=krylovdim,
    gse_cutoff=gse_cutoff,
    gse_maxdim=maxdim_1site,
    label=label_gse_l,
    gse_every_steps=gse_every_steps,
)
metrics_gse_l = tdvp_run_metrics(states_gse_l, physical_sites, H_mpo, x_mpos, z_mpos, exact_densities)
push!(run_series, (; label=label_gse_l, method=:gse10, space=:liouville, metrics=metrics_gse_l))

label_2site_l = "2-TDVP Liouville"
states_2site_l = tdvp_trajectory(
    ρ0_liouville,
    L_mpo,
    dt,
    dt,
    nsteps;
    nsite=2,
    maxdim=maxdim_2site,
    cutoff=cutoff,
    label=label_2site_l,
)
metrics_2site_l = tdvp_run_metrics(states_2site_l, physical_sites, H_mpo, x_mpos, z_mpos, exact_densities)
push!(run_series, (; label=label_2site_l, method=:tdvp2, space=:liouville, metrics=metrics_2site_l))

# ------------------------------------------------------------------------------
# 5. Diagnostics and sanity checks
# ------------------------------------------------------------------------------

print_section("Diagnostics")

for series in run_series
    print_series_diagnostics(series.label, exact, series.metrics)
    println()
end

@assert all(isfinite, exact.sx) && all(isfinite, exact.sz)

# ------------------------------------------------------------------------------
# 6. Plotting and saved outputs
# ------------------------------------------------------------------------------

print_section("Plotting")

plot_title = "Unitary TFIM TDVP (N=$N, J=$J, h=$h, dt=$dt)"
fig_path_x = joinpath(output_dir, "tdvp_tfim_unitary_mx.png")
fig_path_z = joinpath(output_dir, "tdvp_tfim_unitary_mz.png")
fig_path_cons = joinpath(output_dir, "tdvp_tfim_unitary_conserved.png")

plot_observable_comparison(
    fig_path_x, times, exact.sx, run_series, :sx;
    ylabel=L"$\langle \overline{\sigma}_x \rangle (t)$", title=plot_title,
)
plot_observable_comparison(
    fig_path_z, times, exact.sz, run_series, :sz;
    ylabel=L"$\langle \overline{\sigma}_z \rangle (t)$", title=plot_title,
)
plot_unitary_conserved(fig_path_cons, times, exact, run_series; title="Unitary TFIM conserved quantities")

hilbert_title = "Hilbert TFIM TDVP (N=$N, J=$J, h=$h, dt=$dt)"
liouville_title = "Liouville TFIM TDVP (N=$N, J=$J, h=$h, dt=$dt)"
fig_path_h_x = joinpath(output_dir, "tdvp_tfim_unitary_hilbert_dynamics_mx.png")
fig_path_h_energy = joinpath(output_dir, "tdvp_tfim_unitary_hilbert_energy_drift.png")
fig_path_h_rho = joinpath(output_dir, "tdvp_tfim_unitary_hilbert_rho_error.png")
fig_path_l_x = joinpath(output_dir, "tdvp_tfim_unitary_liouville_dynamics_mx.png")
fig_path_l_energy = joinpath(output_dir, "tdvp_tfim_unitary_liouville_energy_drift.png")
fig_path_l_rho = joinpath(output_dir, "tdvp_tfim_unitary_liouville_rho_error.png")

plot_observable_comparison(
    fig_path_h_x, times, exact.sx, run_series, :sx;
    ylabel=L"$\langle \overline{\sigma}_x \rangle (t)$", title=hilbert_title, space=:hilbert,
)
plot_energy_drift_comparison(fig_path_h_energy, times, run_series; title=hilbert_title, space=:hilbert)
plot_rho_error_comparison(fig_path_h_rho, times, run_series; title=hilbert_title, space=:hilbert)
plot_observable_comparison(
    fig_path_l_x, times, exact.sx, run_series, :sx;
    ylabel=L"$\langle \overline{\sigma}_x \rangle (t)$", title=liouville_title, space=:liouville,
)
plot_energy_drift_comparison(fig_path_l_energy, times, run_series; title=liouville_title, space=:liouville)
plot_rho_error_comparison(fig_path_l_rho, times, run_series; title=liouville_title, space=:liouville)

println("Saved figures:")
println("  $fig_path_x")
println("  $fig_path_z")
println("  $fig_path_cons")
println("  $fig_path_h_x")
println("  $fig_path_h_energy")
println("  $fig_path_h_rho")
println("  $fig_path_l_x")
println("  $fig_path_l_energy")
println("  $fig_path_l_rho")

# ------------------------------------------------------------------------------
# 7. Final summary
# ------------------------------------------------------------------------------

print_section("Summary")

println("Completed unitary TFIM TDVP benchmark.")
println()
println("Main outputs:")
println("  figure: $fig_path_x")
println("  figure: $fig_path_z")
println("  figure: $fig_path_cons")
