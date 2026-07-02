# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: scripts/tebd_tfim_unitary.jl
# Contributor: Gauthameshwar S.
#
# Unitary TFIM (N=4): Hilbert- and Liouville-space TEBD vs dense exp(tL).
#
# Run with:
#   julia --project=. scripts/tebd_tfim_unitary.jl

using Printf
using ProcessTensors
using ITensors
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

function dense_liouvillian_matrix(os_H::OpSum, jump_ops, physical_sites, liouv_sites_shared)
    L_mpo = MPO_Liouville(os_H, liouv_sites_shared; jump_ops=jump_ops)
    d = prod(dim.(physical_sites))
    d2 = d * d
    L_dense = zeros(ComplexF64, d2, d2)
    for b in 1:d, a in 1:d
        q = a + (b - 1) * d
        E = zeros(ComplexF64, d, d)
        E[a, b] = 1.0
        basis_q = to_liouville(hilbert_matrix_to_mpo(E, physical_sites); sites=liouv_sites_shared)
        σ_q = apply(L_mpo, basis_q; cutoff=0.0, maxdim=typemax(Int))
        ρ_out = hilbert_mpo_to_dense(to_hilbert(σ_q), physical_sites)
        L_dense[:, q] = vec(ρ_out)
    end
    return L_dense
end

state_to_density_dense(state::AbstractMPS{Hilbert}, physical_sites) =
    hilbert_mpo_to_dense(to_dm(state), physical_sites)
state_to_density_dense(state::AbstractMPS{Liouville}, physical_sites) =
    hilbert_mpo_to_dense(to_hilbert(state), physical_sites)

density_error(ρ::AbstractMatrix, ρ_ref::AbstractMatrix) =
    norm(ρ - ρ_ref) / max(norm(ρ_ref), eps(Float64))

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

function mean_sx_from_density(ρ::AbstractMatrix, x_ops)
    return real(sum(tr(ρ * O) for O in x_ops) / length(x_ops))
end

function exact_sx_trajectory(L_dense, vec0, d, x_ops, times::AbstractVector)
    sx = Float64[]
    for t in times
        push!(sx, mean_sx_from_density(exact_density_at(t, L_dense, vec0, d), x_ops))
    end
    return sx
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

function mean_sx(state::AbstractMPS{Hilbert}, x_mpos)
    return real(sum(inner(state', O, state) for O in x_mpos) / length(x_mpos))
end

function mean_sx(state::AbstractMPS{Liouville}, x_mpos)
    ρ_h = to_hilbert(state)
    s = 0.0
    for O in x_mpos
        ρO = apply(O, ρ_h; alg="naive", truncate=false)
        s += real(tr(ρO))
    end
    return s / length(x_mpos)
end

function exact_density_at(t::Real, L_dense::AbstractMatrix, vec0::AbstractVector, d::Int)
    Lt = ComplexF64.(L_dense)
    v0 = ComplexF64.(vec0)
    vt = iszero(t) ? v0 : exp(t * Lt) * v0
    return reshape(vt, d, d)
end

function run_hilbert_tebd(ψ0, os_H, physical_sites, x_mpos, L_dense, vec0, d, T_max, dt, alg; maxdim, cutoff)
    ψ = copy(ψ0)
    times = Float64[0.0]
    rho_errs = Float64[density_error(state_to_density_dense(ψ, physical_sites), exact_density_at(0.0, L_dense, vec0, d))]
    sx = Float64[mean_sx(ψ, x_mpos)]
    t = 0.0
    elapsed = @elapsed begin
        while t < T_max - 1e-12
            Δt = min(dt, T_max - t)
            ψ = tebd(ψ, os_H, Δt, Δt; maxdim=maxdim, cutoff=cutoff, alg=alg)
            t += Δt
            push!(times, t)
            ρ_ed = exact_density_at(t, L_dense, vec0, d)
            push!(rho_errs, density_error(state_to_density_dense(ψ, physical_sites), ρ_ed))
            push!(sx, mean_sx(ψ, x_mpos))
        end
    end
    return (; times, rho_errs, sx, elapsed, max_bond=maxlinkdim(ψ))
end

function run_liouville_tebd(ρ0_vec, os_H, physical_sites, x_mpos, L_dense, vec0, d, T_max, dt, alg; maxdim, cutoff, jump_ops)
    current = copy(ρ0_vec)
    times = Float64[0.0]
    rho_errs = Float64[density_error(state_to_density_dense(current, physical_sites), exact_density_at(0.0, L_dense, vec0, d))]
    sx = Float64[mean_sx(current, x_mpos)]
    t = 0.0
    elapsed = @elapsed begin
        while t < T_max - 1e-12
            Δt = min(dt, T_max - t)
            current = tebd(
                current,
                os_H,
                Δt,
                Δt;
                jump_ops=jump_ops,
                maxdim=maxdim,
                cutoff=cutoff,
                alg=alg,
            )
            t += Δt
            push!(times, t)
            ρ_ed = exact_density_at(t, L_dense, vec0, d)
            push!(rho_errs, density_error(state_to_density_dense(current, physical_sites), ρ_ed))
            push!(sx, mean_sx(current, x_mpos))
        end
    end
    return (; times, rho_errs, sx, elapsed, max_bond=maxlinkdim(current))
end

function print_tebd_summary(label::AbstractString, n::Int, dt::Real, result)
    @printf("%s TEBD(%d) with dt=%.2f\n", label, n, dt)
    @printf("  Total time taken: %.3f s\n", result.elapsed)
    @printf("  |ρ - ρ_ED|:       %.3e\n", result.rho_errs[end])
    @printf("  max bond dim:     %d\n", result.max_bond)
    println()
end

function save_sx_plot(run_map, t_exact, sx_exact, output_path, representation::AbstractString, T_max, N, J, h)
    tebd_lw, ed_lw = 2.5, 2.5
    fig = Figure(size=(900, 520))
    ax = Axis(
        fig[1, 1];
        xlabel=L"$t$",
        ylabel=L"$\langle \overline{\sigma}_x \rangle (t)$",
        title="$representation TFIM (N=$N, J=$J, h=$h)",
    )
    xlims!(ax, 0, T_max)
    handles, labels = AbstractPlot[], Any[]
    for dt in dt_list
        t1, sx1 = run_map[(1, dt)].times, run_map[(1, dt)].sx
        t2, sx2 = run_map[(2, dt)].times, run_map[(2, dt)].sx
        h1 = lines!(ax, t1, sx1; linestyle=:dashdot, linewidth=tebd_lw)
        h2 = lines!(ax, t2, sx2; linestyle=:solid, linewidth=tebd_lw)
        push!(handles, h1, h2)
        push!(
            labels,
            LaTeXString("\\mathrm{TEBD}(1),\\; dt = $(string(dt))"),
            LaTeXString("\\mathrm{TEBD}(2),\\; dt = $(string(dt))"),
        )
    end
    h_ex = lines!(ax, t_exact, sx_exact; color=:black, linewidth=ed_lw)
    push!(handles, h_ex)
    push!(labels, L"$\mathrm{ED}\,(e^{t L})$")
    axislegend(ax, handles, labels; position=:rt, nbanks=2, fontsize=10)
    save(output_path, fig)
end

function save_rho_error_plot(run_map, output_path, representation::AbstractString, T_max, N, J, h)
    tebd_lw = 2.5
    fig = Figure(size=(900, 520))
    ax = Axis(
        fig[1, 1];
        xlabel=L"$t$",
        ylabel="||ρ - ρ_ED|| / ||ρ_ED||",
        title="$representation TFIM (N=$N, J=$J, h=$h)",
    )
    xlims!(ax, 0, T_max)
    handles, labels = AbstractPlot[], Any[]
    for dt in dt_list
        t1, err1 = run_map[(1, dt)].times, run_map[(1, dt)].rho_errs
        t2, err2 = run_map[(2, dt)].times, run_map[(2, dt)].rho_errs
        h1 = lines!(ax, t1, err1; linestyle=:dashdot, linewidth=tebd_lw)
        h2 = lines!(ax, t2, err2; linestyle=:solid, linewidth=tebd_lw)
        push!(handles, h1, h2)
        push!(
            labels,
            LaTeXString("\\mathrm{TEBD}(1),\\; dt = $(string(dt))"),
            LaTeXString("\\mathrm{TEBD}(2),\\; dt = $(string(dt))"),
        )
    end
    axislegend(ax, handles, labels; position=:rt, nbanks=2, fontsize=10)
    save(output_path, fig)
end

# ------------------------------------------------------------------------------
# 2. Script parameters
# ------------------------------------------------------------------------------

const N = 4
const J = 1.0
const h = 1.2
const T_max = 9.0
const dt_list = Float64[0.2, 0.1, 0.05]
const trotter_orders = (1, 2)
const maxdim = 128
const cutoff = 1e-12
const n_exact = 281

output_dir = joinpath(@__DIR__, "figures")
mkpath(output_dir)

# ------------------------------------------------------------------------------
# 3. Define the physical problem
# ------------------------------------------------------------------------------

print_section("Problem setup")

println("Unitary transverse-field Ising chain: Hilbert- and Liouville-space TEBD vs dense exp(tL).")
println()

@printf("chain length N        = %d\n", N)
@printf("J, h                  = %.3f, %.3f\n", J, h)
@printf("final time T_max      = %.3f\n", T_max)
@printf("TEBD dt values        = %s\n", join(string.(dt_list), ", "))
@printf("Trotter orders        = %s\n", join(string.(trotter_orders), ", "))
@printf("max bond dimension    = %d\n", maxdim)
@printf("SVD cutoff            = %.1e\n", cutoff)

physical_sites = siteinds("S=1/2", N)
liouv_sites_shared = liouv_sites(physical_sites)
os_H = tfim_hamiltonian(N; J=J, h=h)
jump_ops = Tuple{Number, String, Int}[]

ψ0 = MPS(physical_sites, fill("Up", N))
ρ0 = to_dm(ψ0)
ρ0_vec = to_liouville(ρ0; sites=liouv_sites_shared)
x_mpos = single_site_pauli_mpos("X", physical_sites)

# ------------------------------------------------------------------------------
# 4. Main computation
# ------------------------------------------------------------------------------

print_section("Main computation")

println("Building dense Liouvillian reference...")
d = prod(dim.(physical_sites))
vec0 = vec(ComplexF64.(hilbert_mpo_to_dense(ρ0, physical_sites)))
L_dense = dense_liouvillian_matrix(os_H, jump_ops, physical_sites, liouv_sites_shared)
x_ops = [dense_one_site_operator("X", physical_sites, j) for j in 1:N]
t_exact = collect(range(0.0, T_max; length=n_exact))
sx_exact = exact_sx_trajectory(L_dense, vec0, d, x_ops, t_exact)

run_configs = [(n, dt) for n in trotter_orders for dt in dt_list]
n_runs = length(run_configs)

println("Running Hilbert-space TEBD...")
hilbert_runs = Dict{Tuple{Int, Float64}, NamedTuple}()
for (run_idx, (n, dt)) in enumerate(run_configs)
    alg = Trotter{n}()
    update_status(@sprintf("  Hilbert TEBD(%d) dt=%.2f  run %d/%d", n, dt, run_idx, n_runs))
    hilbert_runs[(n, dt)] = run_hilbert_tebd(
        ψ0, os_H, physical_sites, x_mpos, L_dense, vec0, d, T_max, dt, alg;
        maxdim=maxdim, cutoff=cutoff,
    )
    finish_status(@sprintf("  Hilbert TEBD(%d) dt=%.2f complete", n, dt))
end

println("Running Liouville-space TEBD...")
liouville_runs = Dict{Tuple{Int, Float64}, NamedTuple}()
for (run_idx, (n, dt)) in enumerate(run_configs)
    alg = Trotter{n}()
    update_status(@sprintf("  Liouville TEBD(%d) dt=%.2f  run %d/%d", n, dt, run_idx, n_runs))
    liouville_runs[(n, dt)] = run_liouville_tebd(
        ρ0_vec, os_H, physical_sites, x_mpos, L_dense, vec0, d, T_max, dt, alg;
        maxdim=maxdim, cutoff=cutoff, jump_ops=jump_ops,
    )
    finish_status(@sprintf("  Liouville TEBD(%d) dt=%.2f complete", n, dt))
end

# ------------------------------------------------------------------------------
# 5. Diagnostics and sanity checks
# ------------------------------------------------------------------------------

print_section("Diagnostics")

println("Hilbert-space TEBD")
println("------------------")
for (n, dt) in run_configs
    print_tebd_summary("Hilbert", n, dt, hilbert_runs[(n, dt)])
end

println("Liouville-space TEBD")
println("--------------------")
for (n, dt) in run_configs
    print_tebd_summary("Liouville", n, dt, liouville_runs[(n, dt)])
end

@assert all(r -> all(isfinite, r.rho_errs), values(hilbert_runs))
@assert all(r -> all(isfinite, r.rho_errs), values(liouville_runs))

# ------------------------------------------------------------------------------
# 6. Plotting and saved outputs
# ------------------------------------------------------------------------------

print_section("Plotting")

fig_paths = String[]
for (representation, run_map, tag) in (
    ("Hilbert", hilbert_runs, "hilbert"),
    ("Liouville", liouville_runs, "liouville"),
)
    sx_path = joinpath(output_dir, "tebd_tfim_unitary_$(tag)_dynamics_mx.png")
    save_sx_plot(run_map, t_exact, sx_exact, sx_path, representation, T_max, N, J, h)
    push!(fig_paths, sx_path)
    rho_path = joinpath(output_dir, "tebd_tfim_unitary_$(tag)_rho_error.png")
    save_rho_error_plot(run_map, rho_path, representation, T_max, N, J, h)
    push!(fig_paths, rho_path)
end

println("Saved figures:")
for fig_path in fig_paths
    println("  $fig_path")
end

# ------------------------------------------------------------------------------
# 7. Final summary
# ------------------------------------------------------------------------------

print_section("Summary")

println("Completed unitary TFIM TEBD benchmark.")
println()
println("Main outputs:")
for fig_path in fig_paths
    println("  figure: $fig_path")
end
