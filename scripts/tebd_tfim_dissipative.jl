# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: scripts/tebd_tfim_dissipative.jl
# Contributor: Gauthameshwar S.
#
# Dissipative TFIM (N=4): compares Trotter{1,2,4} Liouville TEBD at fixed dt
# against dense exp(tL) for excited-spin density and mean transverse magnetization.
#
# Run with:
#   julia --project=. scripts/tebd_tfim_dissipative.jl

using Printf
using ProcessTensors
using ITensors
using ITensors.Ops: Trotter
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

function excited_density_trace_mpo(ρ_vec::MPS{Liouville}, z_mpos::Vector{MPO{Hilbert}})
    return 0.5 * (1 + mean_pauli_trace_mpo(ρ_vec, z_mpos))
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

function mx_nup_from_vec(v::AbstractVector, d::Int, x_ops, z_ops)
    ρ = reshape(ComplexF64.(v), d, d)
    mean_x = average_observable_dense(ρ, x_ops)
    mean_z = average_observable_dense(ρ, z_ops)
    return mean_x, 0.5 * (1 + mean_z)
end

function exact_observable_trajectory(L_dense, vec0::AbstractVector, d::Int, x_ops, z_ops, times::AbstractVector)
    mean_x_vals, nup_vals = Float64[], Float64[]
    Lt = ComplexF64.(L_dense)
    v0 = ComplexF64.(vec0)
    for t in times
        vt = t == 0 ? v0 : exp(t * Lt) * v0
        mean_x, nup = mx_nup_from_vec(vt, d, x_ops, z_ops)
        push!(mean_x_vals, mean_x)
        push!(nup_vals, nup)
    end
    return mean_x_vals, nup_vals
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
const dt = 0.1
const trotter_orders = (1, 2, 4)
const maxdim = 128
const cutoff = 1e-12
const n_exact = 281

output_dir = joinpath(@__DIR__, "figures")
mkpath(output_dir)

# ------------------------------------------------------------------------------
# 3. Define the physical problem
# ------------------------------------------------------------------------------

print_section("Problem setup")

println("Dissipative TFIM: Trotter-order Liouville TEBD vs dense exp(tL).")
println()

@printf("chain length N        = %d\n", N)
@printf("J, h                  = %.3f, %.3f\n", J, h)
@printf("decay rate γ          = %.3f\n", γ)
@printf("final time T_max      = %.3f\n", T_max)
@printf("TEBD timestep dt      = %.3f\n", dt)
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
steady_nup = 0.5 * (1 + steady_z)

t_exact = collect(range(0.0, T_max; length=n_exact))
mx_exact, nup_exact = exact_observable_trajectory(L_dense, vec0, d, x_ops, z_ops, t_exact)

println("Running Liouville TEBD sweeps...")
tebd_runs = Tuple{Int, Vector{Float64}, Vector{Float64}, Vector{Float64}}[]
n_runs = length(trotter_orders)

for (run_idx, order) in enumerate(trotter_orders)
    alg = Trotter{order}()
    times = Float64[0.0]
    mean_x_vals, nup_vals = Float64[], Float64[]
    current = copy(ρ0_vec)
    push!(mean_x_vals, mean_pauli_trace_mpo(current, x_mpos))
    push!(nup_vals, excited_density_trace_mpo(current, z_mpos))
    t = 0.0
    step = 0
    while t < T_max - 1e-12
        step += 1
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
            progress=false,
        )
        t += Δt
        push!(times, t)
        push!(mean_x_vals, mean_pauli_trace_mpo(current, x_mpos))
        push!(nup_vals, excited_density_trace_mpo(current, z_mpos))
        update_status(
            @sprintf(
                "  TEBD(%d) dt=%.2f  run %d/%d  step %d  t=%.3f",
                order,
                dt,
                run_idx,
                n_runs,
                step,
                t,
            )
        )
    end
    finish_status(@sprintf("  TEBD(%d) dt=%.2f complete (%d time points)", order, dt, length(times)))
    push!(tebd_runs, (order, times, mean_x_vals, nup_vals))
end

# ------------------------------------------------------------------------------
# 5. Diagnostics and sanity checks
# ------------------------------------------------------------------------------

print_section("Diagnostics")

@assert abs(λ0) < 1e-8 "Expected near-zero Liouvillian eigenvalue for steady state, got $λ0"

@printf("ED steady ⟨X̄⟩          = %.6f\n", steady_x)
@printf("ED steady n̄_↑          = %.6f\n", steady_nup)

tebd_errors = map(tebd_runs) do (order, times, mean_x_vals, nup_vals)
    err_x = max_interp_error(t_exact, mx_exact, times, mean_x_vals)
    err_nup = max_interp_error(t_exact, nup_exact, times, nup_vals)
    @printf(
        "TEBD(%d) dt=%.2f  max |⟨X̄⟩−ED| = %.3e  max |n̄_↑−ED| = %.3e\n",
        order,
        dt,
        err_x,
        err_nup,
    )
    if err_x > 0.15 || err_nup > 0.15
        @warn "TEBD observable error is larger than expected." (order=order, dt=dt, err_x=err_x, err_nup=err_nup)
    end
    (err_x, err_nup)
end
max_mx_err = maximum(first, tebd_errors)
max_nup_err = maximum(last, tebd_errors)

@assert all(isfinite, mx_exact) && all(isfinite, nup_exact)

# ------------------------------------------------------------------------------
# 6. Plotting and saved outputs
# ------------------------------------------------------------------------------

print_section("Plotting")

# Draw ED first (behind) with a slightly thicker stroke so TEBD overlays it
# without fully hiding the reference.
ed_lw = 3.8
tebd_lw = 2.2
order_styles = Dict(
    1 => (:dot, :tomato),
    2 => (:dashdot, :dodgerblue),
    4 => (:solid, :goldenrod1),
)

fig_nup = Figure(size=(900, 520))
ax_nup = Axis(
    fig_nup[1, 1];
    xlabel=L"$t$",
    ylabel=L"$\overline{n}_{\uparrow}(t)$",
    title="Dissipative TFIM (N=$N, J=$J, h=$h, γ=$γ, Δt=$dt)",
)
xlims!(ax_nup, 0, T_max)
h_ss_nup = lines!(
    ax_nup,
    [0.0, T_max],
    [steady_nup, steady_nup];
    color=(:gray, 0.85),
    linestyle=:dash,
    linewidth=2.4,
)
h_ex_nup = lines!(ax_nup, t_exact, nup_exact; color=:black, linewidth=ed_lw)
hnup, lnup = AbstractPlot[h_ex_nup, h_ss_nup], Any[L"$\mathrm{ED}\,(e^{t L})$", L"$\mathrm{steady~state}$"]
for (order, times, _, nup_vals) in tebd_runs
    style, color = order_styles[order]
    tebd_line = lines!(ax_nup, times, nup_vals; linestyle=style, color=color, linewidth=tebd_lw)
    push!(hnup, tebd_line)
    push!(lnup, LaTeXString("\\mathrm{TEBD}($(order))"))
end
axislegend(ax_nup, hnup, lnup; position=:rt, fontsize=11)
fig_path_nup = joinpath(output_dir, "tebd_tfim_dissipative_dynamics_nup.png")
save(fig_path_nup, fig_nup)

fig_x = Figure(size=(900, 520))
ax_x = Axis(
    fig_x[1, 1];
    xlabel=L"$t$",
    ylabel=L"$\overline{X}(t)$",
    title="Dissipative TFIM (N=$N, J=$J, h=$h, γ=$γ, Δt=$dt)",
)
xlims!(ax_x, 0, T_max)
h_ss_x = lines!(
    ax_x,
    [0.0, T_max],
    [steady_x, steady_x];
    color=(:gray, 0.85),
    linestyle=:dash,
    linewidth=2.4,
)
h_ex_x = lines!(ax_x, t_exact, mx_exact; color=:black, linewidth=ed_lw)
hx, lx = AbstractPlot[h_ex_x, h_ss_x], Any[L"$\mathrm{ED}\,(e^{t L})$", L"$\mathrm{steady~state}$"]
for (order, times, mean_x_vals, _) in tebd_runs
    style, color = order_styles[order]
    tebd_line = lines!(ax_x, times, mean_x_vals; linestyle=style, color=color, linewidth=tebd_lw)
    push!(hx, tebd_line)
    push!(lx, LaTeXString("\\mathrm{TEBD}($(order))"))
end
axislegend(ax_x, hx, lx; position=:rt, fontsize=11)
fig_path_x = joinpath(output_dir, "tebd_tfim_dissipative_dynamics_mx.png")
save(fig_path_x, fig_x)

println("Saved figures:")
println("  $fig_path_nup")
println("  $fig_path_x")

# ------------------------------------------------------------------------------
# 7. Final summary
# ------------------------------------------------------------------------------

print_section("Summary")

println("Completed dissipative TFIM TEBD Trotter-order benchmark.")
println()
println("Main outputs:")
println("  figure: $fig_path_nup")
println("  figure: $fig_path_x")
println()
println("Main diagnostics:")
@printf("  max |⟨X̄⟩ − ED|      = %.3e\n", max_mx_err)
@printf("  max |n̄_↑ − ED|      = %.3e\n", max_nup_err)
@printf("  ED steady ⟨X̄⟩       = %.6f\n", steady_x)
@printf("  ED steady n̄_↑       = %.6f\n", steady_nup)
