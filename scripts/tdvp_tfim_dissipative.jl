# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: scripts/tdvp_tfim_dissipative.jl
# Contributor: Gauthameshwar S.
#
# Dissipative TFIM (N=4): 1-TDVP + global subspace expansion at different update
# frequencies, together with 2-TDVP and dense exp(tL).
#
# Run with:
#   julia --project=. scripts/tdvp_tfim_dissipative.jl

using Printf
using ProcessTensors
using ITensors
using ITensorMPS: expand, orthogonalize!
using LinearAlgebra
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
    return foldl(kron, local_ops)
end

function average_observable_dense(ρ::AbstractMatrix{<:Number}, embedded_ops)
    return real(sum(tr(ρ * O) for O in embedded_ops) / length(embedded_ops))
end

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

function tdvp_trajectory(ρ0_vec, operator, dt::Float64, nsteps::Int; nsite::Int, maxdim::Int, cutoff::Float64, label::AbstractString)
    states = Vector{typeof(ρ0_vec)}(undef, nsteps + 1)
    states[1] = copy(ρ0_vec)
    current = copy(ρ0_vec)
    progress_stride = max(1, nsteps ÷ 20)
    for step in 1:nsteps
        current = tdvp(operator, dt, current; time_step=dt, nsite=nsite, maxdim=maxdim, cutoff=cutoff, outputlevel=0)
        states[step + 1] = current
        if step == 1 || step == nsteps || step % progress_stride == 0
            update_status(@sprintf("  %-18s step %4d / %4d  t = %.3f  bond = %d", label, step, nsteps, step * dt, maxlinkdim(current)))
        end
    end
    finish_status(@sprintf("  %-18s complete (%d snapshots)", label, length(states)))
    return states
end

function tdvp_run_metrics(states, physical_sites, trace_bra, H_mpo, x_mpos, z_mpos)
    energy, trace_err, herm, min_eig, sx, sz, bond_dims = Float64[], Float64[], Float64[], Float64[], Float64[], Float64[], Int[]
    energy0 = energy_expectation_mpo(first(states), H_mpo)
    for state in states
        ρ_dense = liouville_state_to_dense(state, physical_sites)
        metrics = dense_density_metrics(ρ_dense)
        push!(energy, energy_expectation_mpo(state, H_mpo))
        push!(trace_err, abs(liouville_trace(state, trace_bra) - 1))
        push!(herm, metrics.hermiticity)
        push!(min_eig, metrics.min_eig)
        push!(sx, mean_pauli_trace_mpo(state, x_mpos))
        push!(sz, mean_pauli_trace_mpo(state, z_mpos))
        push!(bond_dims, maxlinkdim(state))
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
    )
end

function liouville_gse_expand(
    ρ_vec::MPS{Liouville},
    operator;
    krylovdim::Int,
    gse_cutoff::Float64,
    gse_maxdim::Int,
)
    # `expand` preserves the represented state but enlarges the MPS gauge basis,
    # allowing subsequent 1-site TDVP steps to move on a larger manifold.
    expanded_core = expand(
        ρ_vec.core,
        operator.core;
        alg="global_krylov",
        krylovdim=krylovdim,
        cutoff=gse_cutoff,
        apply_kwargs=(; maxdim=gse_maxdim),
    )
    orthogonalize!(expanded_core, 1)
    return MPS{Liouville}(expanded_core, ρ_vec.combiners)
end

function tdvp1_gse_trajectory(
    ρ0_vec::MPS{Liouville},
    operator,
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
    states = Vector{typeof(ρ0_vec)}(undef, nsteps + 1)
    states[1] = copy(ρ0_vec)
    current = copy(ρ0_vec)
    progress_stride = max(1, nsteps ÷ 20)
    for step in 1:nsteps
        if step == 1 || (gse_every_steps > 0 && (step - 1) % gse_every_steps == 0)
            current = liouville_gse_expand(
                current,
                operator;
                krylovdim=krylovdim,
                gse_cutoff=gse_cutoff,
                gse_maxdim=gse_maxdim,
            )
        end
        current = tdvp(
            operator,
            dt,
            current;
            time_step=dt,
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

const TDVP1_COLORS = (:dodgerblue1, :dodgerblue3, :dodgerblue4)
const TDVP2_COLOR = :darkorange2
const ED_LINEWIDTH = 3.6
const TDVP2_LINEWIDTH = 2.0
const TDVP1_LINEWIDTH = 2.4

function plot_observable_comparison(path, times, exact_curve, run_series, observable::Symbol; ylabel, title)
    fig = Figure(size=(900, 520))
    ax = Axis(fig[1, 1]; xlabel=L"$t$", ylabel=ylabel, title=title)
    handles, labels = AbstractPlot[], Any[]
    h_ed = lines!(ax, times, exact_curve; color=:black, linewidth=ED_LINEWIDTH)
    push!(handles, h_ed)
    push!(labels, L"$\mathrm{ED}\,(e^{t L})$")
    series_2site = only(filter(series -> series.nsite == 2, run_series))
    h_2site = lines!(
        ax,
        times,
        getfield(series_2site.metrics, observable);
        color=TDVP2_COLOR,
        linestyle=:solid,
        linewidth=TDVP2_LINEWIDTH,
    )
    push!(handles, h_2site)
    push!(labels, series_2site.label)
    for series in filter(series -> series.nsite == 1, run_series)
        h = lines!(
            ax,
            times,
            getfield(series.metrics, observable);
            color=TDVP1_COLORS[series.style_idx],
            linestyle=:dash,
            linewidth=TDVP1_LINEWIDTH,
        )
        push!(handles, h)
        push!(labels, series.label)
    end
    axislegend(ax, handles, labels; position=:rt, nbanks=2, fontsize=10)
    save(path, fig)
    return path
end

function plot_dissipative_conserved(path, times, exact_data, run_series; title)
    fig = Figure(size=(1100, 700))
    ax_trace = Axis(fig[1, 1]; xlabel=L"$t$", ylabel="trace error", title=title, yscale=log10)
    ax_herm = Axis(fig[1, 2]; xlabel=L"$t$", ylabel="Hermiticity defect", yscale=log10)
    ax_psd = Axis(fig[2, 1]; xlabel=L"$t$", ylabel="min eig")
    ax_bond = Axis(fig[2, 2]; xlabel=L"$t$", ylabel="max bond dim")

    exact_trace = lines!(ax_trace, times, max.(exact_data.trace_err, eps(Float64)); color=:black, linewidth=ED_LINEWIDTH)
    lines!(ax_herm, times, max.(exact_data.hermiticity, eps(Float64)); color=:black, linewidth=ED_LINEWIDTH)
    lines!(ax_psd, times, exact_data.min_eig; color=:black, linewidth=ED_LINEWIDTH)

    handles, labels = AbstractPlot[exact_trace], [L"$\mathrm{ED}$"]
    series_2site = only(filter(series -> series.nsite == 2, run_series))
    data_2site = series_2site.metrics
    h_2site = lines!(ax_trace, times, max.(data_2site.trace_err, eps(Float64)); color=TDVP2_COLOR, linestyle=:solid, linewidth=TDVP2_LINEWIDTH)
    lines!(ax_herm, times, max.(data_2site.hermiticity, eps(Float64)); color=TDVP2_COLOR, linestyle=:solid, linewidth=TDVP2_LINEWIDTH)
    lines!(ax_psd, times, data_2site.min_eig; color=TDVP2_COLOR, linestyle=:solid, linewidth=TDVP2_LINEWIDTH)
    lines!(ax_bond, times, data_2site.bond_dims; color=TDVP2_COLOR, linestyle=:solid, linewidth=TDVP2_LINEWIDTH)
    push!(handles, h_2site)
    push!(labels, series_2site.label)
    for series in filter(series -> series.nsite == 1, run_series)
        data = series.metrics
        h = lines!(ax_trace, times, max.(data.trace_err, eps(Float64)); color=TDVP1_COLORS[series.style_idx], linestyle=:dash, linewidth=TDVP1_LINEWIDTH)
        lines!(ax_herm, times, max.(data.hermiticity, eps(Float64)); color=TDVP1_COLORS[series.style_idx], linestyle=:dash, linewidth=TDVP1_LINEWIDTH)
        lines!(ax_psd, times, data.min_eig; color=TDVP1_COLORS[series.style_idx], linestyle=:dash, linewidth=TDVP1_LINEWIDTH)
        lines!(ax_bond, times, data.bond_dims; color=TDVP1_COLORS[series.style_idx], linestyle=:dash, linewidth=TDVP1_LINEWIDTH)
        push!(handles, h)
        push!(labels, series.label)
    end
    Legend(fig[3, 1:2], handles, labels; orientation=:horizontal, nbanks=2, tellwidth=false, tellheight=true)
    save(path, fig)
    return path
end

# ------------------------------------------------------------------------------
# 2. Script parameters
# ------------------------------------------------------------------------------

const N = 4
const J = 1.0
const h = 1.2
const γ = 0.5
const T_max = 9.0
const dt = 0.05
const nsteps = round(Int, T_max / dt)
const maxdim_1site = 50
const maxdim_2site = 50
const gse_every_steps_list = [1, 10]
const krylovdim = 2
const gse_cutoff = 1e-8
const cutoff = 1e-10

output_dir = joinpath(@__DIR__, "figures")
mkpath(output_dir)

# ------------------------------------------------------------------------------
# 3. Define the physical problem
# ------------------------------------------------------------------------------

print_section("Problem setup")

println("Dissipative transverse-field Ising chain: Liouville TDVP vs dense exp(tL).")
println()

@printf("chain length N        = %d\n", N)
@printf("J, h                  = %.3f, %.3f\n", J, h)
@printf("decay rate γ          = %.3f\n", γ)
@printf("final time T_max      = %.3f\n", T_max)
@printf("time step dt          = %.3f\n", dt)
@printf("1-TDVP max bond dim   = %d\n", maxdim_1site)
@printf("2-TDVP max bond dim   = %d\n", maxdim_2site)
@printf("GSE every-n steps     = %s\n", join(string.(gse_every_steps_list), ", "))
@printf("GSE Krylov dim        = %d\n", krylovdim)
@printf("GSE cutoff            = %.1e\n", gse_cutoff)
@printf("SVD cutoff            = %.1e\n", cutoff)

physical_sites = siteinds("S=1/2", N)
liouv_sites_shared = liouv_sites(physical_sites)
trace_bra = vectorized_identity_state(physical_sites, liouv_sites_shared)
os_H = tfim_hamiltonian(N; J=J, h=h)
H_mpo = MPO(os_H, physical_sites)
H_dense = dense_hamiltonian_matrix(os_H, physical_sites)
jump_ops = tfim_decay_jump_ops(N; γ=γ)

ψ0 = MPS(physical_sites, fill("Up", N))
ρ0 = to_dm(ψ0)
ρ0_vec = to_liouville(ρ0; sites=liouv_sites_shared)
d = prod(dim.(physical_sites))
vec0 = vec(ComplexF64.(hilbert_mpo_to_dense(ρ0, physical_sites)))

x_ops = [dense_one_site_operator("X", physical_sites, j) for j in 1:N]
z_ops = [dense_one_site_operator("Z", physical_sites, j) for j in 1:N]
x_mpos = single_site_pauli_mpos("X", physical_sites)
z_mpos = single_site_pauli_mpos("Z", physical_sites)

L_mpo = liouvillian_mpo(os_H, liouv_sites_shared; jump_ops=jump_ops)
L_dense = dense_liouvillian_matrix(os_H, jump_ops, physical_sites, liouv_sites_shared)
times = collect(range(0.0, step=dt, length=nsteps + 1))

# ------------------------------------------------------------------------------
# 4. Main computation
# ------------------------------------------------------------------------------

print_section("Main computation")

println("Building dense Liouvillian reference...")
exact_densities = exact_density_trajectory(L_dense, vec0, d, times)
exact = exact_metrics(exact_densities, H_dense, x_ops, z_ops)

println("Running Liouville TDVP sweeps...")
run_series = NamedTuple{(:label, :nsite, :style_idx, :metrics), Tuple{String, Int, Int, Any}}[]

label_plain = "1-TDVP plain (χ=$maxdim_1site)"
states_plain = tdvp_trajectory(
    ρ0_vec,
    L_mpo,
    dt,
    nsteps;
    nsite=1,
    maxdim=maxdim_1site,
    cutoff=cutoff,
    label=label_plain,
)
metrics_plain = tdvp_run_metrics(states_plain, physical_sites, trace_bra, H_mpo, x_mpos, z_mpos)
push!(run_series, (; label=label_plain, nsite=1, style_idx=1, metrics=metrics_plain))

for (offset, gse_every_steps) in enumerate(gse_every_steps_list)
    label = "1-TDVP+GSE (every $gse_every_steps)"
    states = tdvp1_gse_trajectory(
        ρ0_vec,
        L_mpo,
        dt,
        nsteps;
        maxdim=maxdim_1site,
        cutoff=cutoff,
        krylovdim=krylovdim,
        gse_cutoff=gse_cutoff,
        gse_maxdim=maxdim_1site,
        gse_every_steps=gse_every_steps,
        label=label,
    )
    metrics = tdvp_run_metrics(states, physical_sites, trace_bra, H_mpo, x_mpos, z_mpos)
    push!(run_series, (; label, nsite=1, style_idx=offset + 1, metrics))
end

label_2site = "2-TDVP (χ=$maxdim_2site)"
states_2site = tdvp_trajectory(
    ρ0_vec,
    L_mpo,
    dt,
    nsteps;
    nsite=2,
    maxdim=maxdim_2site,
    cutoff=cutoff,
    label=label_2site,
)
metrics_2site = tdvp_run_metrics(states_2site, physical_sites, trace_bra, H_mpo, x_mpos, z_mpos)
push!(run_series, (; label=label_2site, nsite=2, style_idx=0, metrics=metrics_2site))

# ------------------------------------------------------------------------------
# 5. Diagnostics and sanity checks
# ------------------------------------------------------------------------------

print_section("Diagnostics")

observable_errors = map(run_series) do series
    err_x = max_curve_error(exact.sx, series.metrics.sx)
    err_z = max_curve_error(exact.sz, series.metrics.sz)
    @printf("%-18s  max |⟨σ_x⟩−ED| = %.3e  max |⟨σ_z⟩−ED| = %.3e\n", series.label, err_x, err_z)
    (series.label, err_x, err_z)
end

max_sx_err = maximum(last -> last[2], observable_errors)
max_sz_err = maximum(last -> last[3], observable_errors)
max_trace_err = maximum(series -> maximum(series.metrics.trace_err), run_series)

@assert all(isfinite, exact.sx) && all(isfinite, exact.sz)
@printf("max trace error (TDVP) = %.3e\n", max_trace_err)

# ------------------------------------------------------------------------------
# 6. Plotting and saved outputs
# ------------------------------------------------------------------------------

print_section("Plotting")

plot_title = "Dissipative TFIM TDVP (N=$N, J=$J, h=$h, γ=$γ, dt=$dt)"
fig_path_x = joinpath(output_dir, "tdvp_tfim_dissipative_mx.png")
fig_path_z = joinpath(output_dir, "tdvp_tfim_dissipative_mz.png")
fig_path_cons = joinpath(output_dir, "tdvp_tfim_dissipative_conserved.png")

plot_observable_comparison(
    fig_path_x, times, exact.sx, run_series, :sx;
    ylabel=L"$\langle \overline{\sigma}_x \rangle (t)$", title=plot_title,
)
plot_observable_comparison(
    fig_path_z, times, exact.sz, run_series, :sz;
    ylabel=L"$\langle \overline{\sigma}_z \rangle (t)$", title=plot_title,
)
plot_dissipative_conserved(fig_path_cons, times, exact, run_series; title="Dissipative TFIM physicality diagnostics")

println("Saved figures:")
println("  $fig_path_x")
println("  $fig_path_z")
println("  $fig_path_cons")

# ------------------------------------------------------------------------------
# 7. Final summary
# ------------------------------------------------------------------------------

print_section("Summary")

println("Completed dissipative TFIM TDVP benchmark.")
println()
println("Main outputs:")
println("  figure: $fig_path_x")
println("  figure: $fig_path_z")
println("  figure: $fig_path_cons")
println()
println("Main diagnostics:")
@printf("  max |⟨σ_x⟩ − ED|     = %.3e\n", max_sx_err)
@printf("  max |⟨σ_z⟩ − ED|     = %.3e\n", max_sz_err)
@printf("  max trace error      = %.3e\n", max_trace_err)
