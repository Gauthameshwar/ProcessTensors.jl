# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: scripts/tebd_tfim_dissipative.jl
# Contributor: Gauthameshwar S.
#
# Dissipative TFIM (N=4): mean ⟨σ_x⟩, ⟨σ_z⟩ from Liouville TEBD vs dense exp(tL).
#
# Run with:
#   julia --project=. scripts/tebd_tfim_dissipative.jl

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

function dense_steady_state(L_dense::AbstractMatrix{<:Number})
    spec = eigen(ComplexF64.(L_dense))
    idx = argmin(abs.(spec.values))
    d = isqrt(size(L_dense, 1))
    ρ_ss = reshape(spec.vectors[:, idx], d, d)
    ρ_ss ./= tr(ρ_ss)
    ρ_ss = (ρ_ss + ρ_ss') / 2
    ρ_ss ./= tr(ρ_ss)
    return ρ_ss, spec.values[idx]
end

function sx_sz_from_vec(v::AbstractVector, d::Int, x_ops, z_ops)
    ρ = reshape(ComplexF64.(v), d, d)
    return average_observable_dense(ρ, x_ops), average_observable_dense(ρ, z_ops)
end

function exact_observable_trajectory(L_dense, vec0::AbstractVector, d::Int, x_ops, z_ops, times::AbstractVector)
    sx, sz = Float64[], Float64[]
    Lt = ComplexF64.(L_dense)
    v0 = ComplexF64.(vec0)
    for t in times
        vt = t == 0 ? v0 : exp(t * Lt) * v0
        sxi, szi = sx_sz_from_vec(vt, d, x_ops, z_ops)
        push!(sx, sxi)
        push!(sz, szi)
    end
    return sx, sz
end

function interpolate_at(times::AbstractVector, values::AbstractVector, t_query::Real)
    t_query <= times[1] && return values[1]
    t_query >= times[end] && return values[end]
    for k in 1:(length(times) - 1)
        if times[k] <= t_query <= times[k + 1]
            α = (t_query - times[k]) / (times[k + 1] - times[k])
            return (1 - α) * values[k] + α * values[k + 1]
        end
    end
    return values[end]
end

function max_interp_error(t_ref, y_ref, t_cmp, y_cmp)
    err = 0.0
    for (t, y) in zip(t_ref, y_ref)
        y_cmp_at_t = interpolate_at(t_cmp, y_cmp, t)
        err = max(err, abs(y - y_cmp_at_t))
    end
    return err
end

# ------------------------------------------------------------------------------
# 2. Script parameters
# ------------------------------------------------------------------------------

const N = 4
const J = 1.0
const h = 1.2
const γ = 0.5
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

println("Dissipative transverse-field Ising chain: Liouville TEBD vs dense exp(tL).")
println()

@printf("chain length N        = %d\n", N)
@printf("J, h                  = %.3f, %.3f\n", J, h)
@printf("decay rate γ          = %.3f\n", γ)
@printf("final time T_max      = %.3f\n", T_max)
@printf("TEBD dt values        = %s\n", join(string.(dt_list), ", "))
@printf("Trotter orders        = %s\n", join(string.(trotter_orders), ", "))
@printf("max bond dimension    = %d\n", maxdim)
@printf("SVD cutoff            = %.1e\n", cutoff)

physical_sites = siteinds("S=1/2", N)
liouv_sites_shared = liouv_sites(physical_sites)
os_H = tfim_hamiltonian(N; J=J, h=h)
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

# ------------------------------------------------------------------------------
# 4. Main computation
# ------------------------------------------------------------------------------

print_section("Main computation")

println("Building dense Liouvillian reference...")
L_dense = dense_liouvillian_matrix(os_H, jump_ops, physical_sites, liouv_sites_shared)
ρ_ss_exact, λ0 = dense_steady_state(L_dense)
steady_x = average_observable_dense(ρ_ss_exact, x_ops)
steady_z = average_observable_dense(ρ_ss_exact, z_ops)

t_exact = collect(range(0.0, T_max; length=n_exact))
sx_exact, sz_exact = exact_observable_trajectory(L_dense, vec0, d, x_ops, z_ops, t_exact)

println("Running Liouville TEBD sweeps...")
tebd_runs = Tuple{Int, Float64, Vector{Float64}, Vector{Float64}, Vector{Float64}}[]
run_configs = [(n, dt) for n in trotter_orders for dt in dt_list]
n_runs = length(run_configs)

for (run_idx, (n, dt)) in enumerate(run_configs)
    alg = Trotter{n}()
    times = Float64[0.0]
    sx, sz = Float64[], Float64[]
    current = copy(ρ0_vec)
    push!(sx, mean_pauli_trace_mpo(current, x_mpos))
    push!(sz, mean_pauli_trace_mpo(current, z_mpos))
    t = 0.0
    step = 0
    while t < T_max - 1e-12
        step += 1
        Δt = min(dt, T_max - t)
        current = tebd(current, os_H, Δt, Δt; jump_ops=jump_ops, maxdim=maxdim, cutoff=cutoff, alg=alg)
        t += Δt
        push!(times, t)
        push!(sx, mean_pauli_trace_mpo(current, x_mpos))
        push!(sz, mean_pauli_trace_mpo(current, z_mpos))
        update_status(@sprintf("  TEBD(%d) dt=%.2f  run %d/%d  step %d  t=%.3f", n, dt, run_idx, n_runs, step, t))
    end
    finish_status(@sprintf("  TEBD(%d) dt=%.2f complete (%d time points)", n, dt, length(times)))
    push!(tebd_runs, (n, dt, times, sx, sz))
end

# ------------------------------------------------------------------------------
# 5. Diagnostics and sanity checks
# ------------------------------------------------------------------------------

print_section("Diagnostics")

@assert abs(λ0) < 1e-8 "Expected near-zero Liouvillian eigenvalue for steady state, got $λ0"

@printf("ED steady ⟨σ_x⟩         = %.6f\n", steady_x)
@printf("ED steady ⟨σ_z⟩         = %.6f\n", steady_z)

tebd_errors = map(tebd_runs) do (n, dt, times, sx, sz)
    err_x = max_interp_error(t_exact, sx_exact, times, sx)
    err_z = max_interp_error(t_exact, sz_exact, times, sz)
    @printf("TEBD(%d) dt=%.2f  max |⟨σ_x⟩−ED| = %.3e  max |⟨σ_z⟩−ED| = %.3e\n", n, dt, err_x, err_z)
    if err_x > 0.15 || err_z > 0.15
        @warn "TEBD observable error is larger than expected." (order=n, dt=dt, err_x=err_x, err_z=err_z)
    end
    (err_x, err_z)
end
max_sx_err = maximum(first, tebd_errors)
max_sz_err = maximum(last, tebd_errors)

@assert all(isfinite, sx_exact) && all(isfinite, sz_exact)

# ------------------------------------------------------------------------------
# 6. Plotting and saved outputs
# ------------------------------------------------------------------------------

print_section("Plotting")

run_map = Dict((o, dt) => (t, sx, sz) for (o, dt, t, sx, sz) in tebd_runs)
tebd_lw, ed_lw = 2.5, 2.5

fig_x = Figure(size=(900, 520))
ax_x = Axis(
    fig_x[1, 1];
    xlabel=L"$t$",
    ylabel=L"$\langle \overline{\sigma}_x \rangle (t)$",
    title="Dissipative TFIM (N=$N, J=$J, h=$h, γ=$γ)",
)
xlims!(ax_x, 0, T_max)
h_ss_x = lines!(ax_x, [0.0, T_max], [steady_x, steady_x]; color=(:gray, 0.85), linestyle=:dash, linewidth=2.4)
hx, lx = AbstractPlot[], Any[]
tebd_x = Tuple{AbstractPlot, AbstractPlot}[]
for dt in dt_list
    t1, sx1, _ = run_map[(1, dt)]
    t2, sx2, _ = run_map[(2, dt)]
    h1 = lines!(ax_x, t1, sx1; linestyle=:dashdot, linewidth=tebd_lw)
    h2 = lines!(ax_x, t2, sx2; linestyle=:solid, linewidth=tebd_lw)
    push!(tebd_x, (h1, h2))
end
h_ex_x = lines!(ax_x, t_exact, sx_exact; color=:black, linewidth=ed_lw)
push!(hx, h_ex_x, h_ss_x)
push!(lx, L"$\mathrm{ED}\,(e^{t L})$", L"$\mathrm{steady~state}$")
for (i, dt) in enumerate(dt_list)
    h1, h2 = tebd_x[i]
    push!(hx, h1, h2)
    push!(lx, LaTeXString("\\mathrm{TEBD}(1),\\; dt = $(string(dt))"), LaTeXString("\\mathrm{TEBD}(2),\\; dt = $(string(dt))"))
end
axislegend(ax_x, hx, lx; position=:rt, nbanks=2, fontsize=10)
fig_path_x = joinpath(output_dir, "tebd_tfim_dissipative_dynamics_mx.png")
save(fig_path_x, fig_x)

fig_z = Figure(size=(900, 520))
ax_z = Axis(
    fig_z[1, 1];
    xlabel=L"$t$",
    ylabel=L"$\langle \overline{\sigma}_z \rangle (t)$",
    title="Dissipative TFIM (N=$N, J=$J, h=$h, γ=$γ)",
)
xlims!(ax_z, 0, T_max)
h_ss_z = lines!(ax_z, [0.0, T_max], [steady_z, steady_z]; color=(:gray, 0.85), linestyle=:dash, linewidth=2.4)
hz, lz = AbstractPlot[], Any[]
tebd_z = Tuple{AbstractPlot, AbstractPlot}[]
for dt in dt_list
    t1, _, sz1 = run_map[(1, dt)]
    t2, _, sz2 = run_map[(2, dt)]
    h1 = lines!(ax_z, t1, sz1; linestyle=:dashdot, linewidth=tebd_lw)
    h2 = lines!(ax_z, t2, sz2; linestyle=:solid, linewidth=tebd_lw)
    push!(tebd_z, (h1, h2))
end
h_ex_z = lines!(ax_z, t_exact, sz_exact; color=:black, linewidth=ed_lw)
push!(hz, h_ex_z, h_ss_z)
push!(lz, L"$\mathrm{ED}\,(e^{t L})$", L"$\mathrm{steady~state}$")
for (i, dt) in enumerate(dt_list)
    h1, h2 = tebd_z[i]
    push!(hz, h1, h2)
    push!(lz, LaTeXString("\\mathrm{TEBD}(1),\\; dt = $(string(dt))"), LaTeXString("\\mathrm{TEBD}(2),\\; dt = $(string(dt))"))
end
axislegend(ax_z, hz, lz; position=:rt, nbanks=2, fontsize=10)
fig_path_z = joinpath(output_dir, "tebd_tfim_dissipative_dynamics_mz.png")
save(fig_path_z, fig_z)

println("Saved figures:")
println("  $fig_path_x")
println("  $fig_path_z")

# ------------------------------------------------------------------------------
# 7. Final summary
# ------------------------------------------------------------------------------

print_section("Summary")

println("Completed dissipative TFIM TEBD benchmark.")
println()
println("Main outputs:")
println("  figure: $fig_path_x")
println("  figure: $fig_path_z")
println()
println("Main diagnostics:")
@printf("  max |⟨σ_x⟩ − ED|     = %.3e\n", max_sx_err)
@printf("  max |⟨σ_z⟩ − ED|     = %.3e\n", max_sz_err)
@printf("  ED steady ⟨σ_x⟩      = %.6f\n", steady_x)
@printf("  ED steady ⟨σ_z⟩      = %.6f\n", steady_z)
