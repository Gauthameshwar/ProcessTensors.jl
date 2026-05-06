# Dissipative TFIM (N=4): time dynamics of mean ⟨σ_x⟩, ⟨σ_z⟩.
# Compares exact open-system evolution vec(ρ(t)) = exp(t L) vec(ρ(0)) to Liouville TEBD
# for several (Trotter order, dt). Horizontal dashed lines: ED steady-state observables.
#
# Run from repo root:
#   julia --project=. scripts/tebd_tfim.jl
#
# Outputs (next to this file):
#   tebd_tfim_dynamics_mx.png
#   tebd_tfim_dynamics_mz.png

using ProcessTensors, ITensors, LinearAlgebra
using CairoMakie, LaTeXStrings

const _ROOT = @__DIR__
const _REPO_ROOT = joinpath(_ROOT, "..")

# Needed to import the dense constructions of the Liouvillian and the steady state, and the single-site Pauli MPOs.
include(joinpath(_REPO_ROOT, "test", "time_evolution", "tebd_test_utils.jl"))

# --- TFIM model -------------------------------------------------------------------------

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
    Ns = length(embedded_ops)
    return real(sum(LinearAlgebra.tr(ρ * O) for O in embedded_ops) / Ns)
end

function dense_steady_state(L_dense::AbstractMatrix{<:Number})
    spec = eigen(ComplexF64.(L_dense))
    idx = argmin(abs.(spec.values))
    d = isqrt(size(L_dense, 1))
    ρ_ss = reshape(spec.vectors[:, idx], d, d)
    ρ_ss ./= LinearAlgebra.tr(ρ_ss)
    ρ_ss = (ρ_ss + ρ_ss') / 2
    ρ_ss ./= LinearAlgebra.tr(ρ_ss)
    return ρ_ss, spec.values[idx]
end

function sx_sz_from_vec(v::AbstractVector, d::Int, x_ops, z_ops)
    ρ = reshape(ComplexF64.(v), d, d)
    return average_observable_dense(ρ, x_ops), average_observable_dense(ρ, z_ops)
end

# Exact ρ(t) in vectorized form: vec(ρ(t)) = exp(t L) vec(ρ(0)).
function exact_observable_trajectory(L_dense, vec0::AbstractVector, d::Int, x_ops, z_ops, times::AbstractVector)
    sx = Float64[]
    sz = Float64[]
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

"""
TEBD trajectory: repeated steps of size at most `dt`, stopping exactly at `T_max`.
Records mean ⟨σ_x⟩, ⟨σ_z⟩ after each completed segment using `to_hilbert` + `apply` + `tr`
"""
function tebd_observable_trajectory(
    ρ0_vec,
    os_H::OpSum,
    jump_ops,
    physical_sites,
    x_mpos::Vector{MPO{Hilbert}},
    z_mpos::Vector{MPO{Hilbert}},
    T_max::Float64,
    dt::Float64,
    order::Int;
    maxdim::Int,
    cutoff::Float64,
)
    times = Float64[0.0]
    sx = Float64[]
    sz = Float64[]
    current = copy(ρ0_vec)
    push!(sx, mean_pauli_trace_mpo(current, x_mpos))
    push!(sz, mean_pauli_trace_mpo(current, z_mpos))
    t = 0.0
    while t < T_max - 1e-12
        Δt = min(dt, T_max - t)
        current = tebd(current, os_H, Δt, Δt; jump_ops=jump_ops, maxdim=maxdim, cutoff=cutoff, order=order)
        t += Δt
        push!(times, t)
        push!(sx, mean_pauli_trace_mpo(current, x_mpos))
        push!(sz, mean_pauli_trace_mpo(current, z_mpos))
    end
    return times, sx, sz
end

# --- main -------------------------------------------------------------------------------

function main()
    println("=== Dissipative TFIM: ⟨σ_x⟩(t), ⟨σ_z⟩(t) — exact exp(tL) vs TEBD ===")

    N = 4
    T_max = 9.0
    physical_sites = siteinds("S=1/2", N)
    liouv_sites_shared = liouv_sites(physical_sites)

    os_H = tfim_hamiltonian(N; J=1.0, h=1.2)
    jump_ops = tfim_decay_jump_ops(N; γ=0.5)

    ψ0 = MPS(physical_sites, fill("Up", N))
    ρ0 = to_dm(ψ0)
    ρ0_vec = to_liouville(ρ0; sites=liouv_sites_shared)

    L_dense = dense_liouvillian_matrix(os_H, jump_ops, physical_sites, liouv_sites_shared)
    ρ_ss_exact, λ0 = dense_steady_state(L_dense)
    @assert abs(λ0) < 1e-8 "Expected near-zero Liouvillian eigenvalue for steady state, got $λ0"

    x_ops = [dense_one_site_operator("X", physical_sites, j) for j in 1:N]
    z_ops = [dense_one_site_operator("Z", physical_sites, j) for j in 1:N]
    x_mpos = single_site_pauli_mpos("X", physical_sites)
    z_mpos = single_site_pauli_mpos("Z", physical_sites)
    steady_x = average_observable_dense(ρ_ss_exact, x_ops)
    steady_z = average_observable_dense(ρ_ss_exact, z_ops)
    println("ED steady state: mean ⟨σ_x⟩ = ", steady_x, ", mean ⟨σ_z⟩ = ", steady_z)

    d = prod(dim.(physical_sites))
    vec0 = vec(ComplexF64.(hilbert_mpo_to_dense(ρ0, physical_sites)))

    # Fine time grid for the exact reference curve
    t_exact = collect(range(0.0, T_max; length=281))
    sx_exact, sz_exact = exact_observable_trajectory(L_dense, vec0, d, x_ops, z_ops, t_exact)

    dt_list = Float64[0.2, 0.1, 0.05]
    orders = [1, 2]
    maxdim = 128
    cutoff = 1e-12

    tebd_runs = Tuple{Int, Float64, Vector{Float64}, Vector{Float64}, Vector{Float64}}[]
    for order in orders
        for dt in dt_list
            times, sx, sz = tebd_observable_trajectory(
                ρ0_vec,
                os_H,
                jump_ops,
                physical_sites,
                x_mpos,
                z_mpos,
                T_max,
                dt,
                order;
                maxdim=maxdim,
                cutoff=cutoff,
            )
            push!(tebd_runs, (order, dt, times, sx, sz))
            println("  TEBD order=$order dt=$dt : $(length(times)) points, final t=$(last(times))")
        end
    end

    run_map = Dict((o, dt) => (t, sx, sz) for (o, dt, t, sx, sz) in tebd_runs)
    tebd_lw = 2.5
    ed_lw = 2.5

    # --- ⟨σ_x⟩(t) ---
    fig_x = Figure(size=(900, 520))
    ax_x = Axis(
        fig_x[1, 1];
        xlabel=L"$t$",
        ylabel=L"$\langle \overline{\sigma}_x \rangle (t)$",
        title="Dissipative TFIM (N=$N, J=$J, h=$h)",
    )
    xlims!(ax_x, 0, T_max)
    h_ss_x = lines!(ax_x, [0.0, T_max], [steady_x, steady_x]; color=(:gray, 0.85), linestyle=:dash, linewidth=2.4)
    hx = AbstractPlot[]
    lx = Any[]
    tebd_x = Tuple{AbstractPlot, AbstractPlot}[]
    for (_, dt) in enumerate(dt_list)
        t1, sx1, _ = run_map[(1, dt)]
        t2, sx2, _ = run_map[(2, dt)]
        h1 = lines!(ax_x, t1, sx1; linestyle=:dashdot, linewidth=tebd_lw)
        h2 = lines!(ax_x, t2, sx2; linestyle=:solid, linewidth=tebd_lw)
        push!(tebd_x, (h1, h2))
    end
    h_ex_x = lines!(ax_x, t_exact, sx_exact; color=:black, linewidth=ed_lw)
    push!(hx, h_ex_x)
    push!(lx, L"$\mathrm{ED}\,(e^{t L})$")
    push!(hx, h_ss_x)
    push!(lx, L"$\mathrm{steady~state}$")
    for (i, dt) in enumerate(dt_list)
        h1, h2 = tebd_x[i]
        push!(hx, h1)
        push!(hx, h2)
        push!(lx, LaTeXString("\\mathrm{TEBD}(1),\\; dt = $(string(dt))"))
        push!(lx, LaTeXString("\\mathrm{TEBD}(2),\\; dt = $(string(dt))"))
    end
    axislegend(ax_x, hx, lx; position=:rt, nbanks=2, fontsize=10)
    out_x = joinpath(_ROOT, "tebd_tfim_dynamics_mx.png")
    save(out_x, fig_x)
    println("Saved: ", out_x)

    # --- ⟨σ_z⟩(t) ---
    fig_z = Figure(size=(900, 520))
    ax_z = Axis(
        fig_z[1, 1];
        xlabel=L"$t$",
        ylabel=L"$\langle \overline{\sigma}_z \rangle (t)$",
        title="Dissipative TFIM (N=$N, J=$J, h=$h)",
    )
    xlims!(ax_z, 0, T_max)
    h_ss_z = lines!(ax_z, [0.0, T_max], [steady_z, steady_z]; color=(:gray, 0.85), linestyle=:dash, linewidth=2.4)
    hz = AbstractPlot[]
    lz = Any[]
    tebd_z = Tuple{AbstractPlot, AbstractPlot}[]
    for (_, dt) in enumerate(dt_list)
        t1, _, sz1 = run_map[(1, dt)]
        t2, _, sz2 = run_map[(2, dt)]
        h1 = lines!(ax_z, t1, sz1; linestyle=:dashdot, linewidth=tebd_lw)
        h2 = lines!(ax_z, t2, sz2; linestyle=:solid, linewidth=tebd_lw)
        push!(tebd_z, (h1, h2))
    end
    h_ex_z = lines!(ax_z, t_exact, sz_exact; color=:black, linewidth=ed_lw)
    push!(hz, h_ex_z)
    push!(lz, L"$\mathrm{ED}\,(e^{t L})$")
    push!(hz, h_ss_z)
    push!(lz, L"$\mathrm{steady~state}$")
    for (i, dt) in enumerate(dt_list)
        h1, h2 = tebd_z[i]
        push!(hz, h1)
        push!(hz, h2)
        push!(lz, LaTeXString("\\mathrm{TEBD}(1),\\; dt = $(string(dt))"))
        push!(lz, LaTeXString("\\mathrm{TEBD}(2),\\; dt = $(string(dt))"))
    end
    axislegend(ax_z, hz, lz; position=:rt, nbanks=2, fontsize=10)
    out_z = joinpath(_ROOT, "tebd_tfim_dynamics_mz.png")
    save(out_z, fig_z)
    println("Saved: ", out_z)

    println("✓ Done")
end

main()
