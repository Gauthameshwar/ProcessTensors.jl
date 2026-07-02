# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: scripts/pt_tfim_multimode.jl
# Contributor: Gauthameshwar S.
#
# Multimode spin-bath process tensor: fused-memory `evolve` vs joint Liouville ED.
#
# Run with:
#   julia --project=. scripts/pt_tfim_multimode.jl

import Pkg

const REPO_ROOT = dirname(@__DIR__)
const _PLOT_ENV = joinpath(@__DIR__, ".plot_examples_env")

function activate_plot_examples_env!()
    mkpath(_PLOT_ENV)
    Pkg.activate(_PLOT_ENV)
    manifest = joinpath(_PLOT_ENV, "Manifest.toml")
    if !isfile(manifest)
        Pkg.develop(Pkg.PackageSpec(path=REPO_ROOT))
        Pkg.add([Pkg.PackageSpec(name="CairoMakie"), Pkg.PackageSpec(name="LaTeXStrings")])
    else
        Pkg.instantiate()
    end
    return nothing
end

activate_plot_examples_env!()

using Printf
using CairoMakie
using LaTeXStrings
using ITensors
using LinearAlgebra
using ITensors.Ops: Exact
using ProcessTensors

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

function reduced_system_ρ(state_l, dsys)
    rho_h = to_hilbert(state_l)
    sites = [
        only(filter(i -> plev(i) == 0 && hastags(i, "Site"), inds(rho_h.core[j])))
        for j in eachindex(rho_h.core)
    ]
    T = foldl(*, rho_h)
    A = Array(T, prime.(sites)..., sites...)
    return reshape(ComplexF64.(A), dsys, dsys)
end

function partial_trace_system(rho_h, dsys, denv)
    sites = [
        only(filter(i -> plev(i) == 0 && hastags(i, "Site"), inds(rho_h.core[j])))
        for j in eachindex(rho_h.core)
    ]
    T = foldl(*, rho_h)
    A = Array(T, prime.(sites)..., sites...)
    ρ4 = reshape(ComplexF64.(A), dsys, denv, dsys, denv)
    ρ_red = zeros(ComplexF64, dsys, dsys)
    for e in 1:denv
        ρ_red .+= @view ρ4[:, e, :, e]
    end
    return ρ_red
end

function pauli_expectations(ρ::AbstractMatrix{<:Number}, σx, σy, σz)
    return real(tr(ρ * σx)), real(tr(ρ * σy)), real(tr(ρ * σz))
end

# ------------------------------------------------------------------------------
# 2. Script parameters
# ------------------------------------------------------------------------------

const joint_ed_frob_tol = 0.05  # matches test/process_tensor/test_ed_multimode_spin.jl

const dt = 0.1
const nsteps = 24
const nmodes = 4
const final_time = dt * nsteps

output_dir = joinpath(@__DIR__, "figures")
mkpath(output_dir)

σx = ComplexF64[0 1; 1 0]
σy = ComplexF64[0 -im; im 0]
σz = ComplexF64[1 0; 0 -1]
dsys = 2
denv = 2^nmodes

mode_w = [0.5 + 0.1 * m for m in 1:nmodes]
mode_g = [0.2 + 0.3 * m for m in 1:nmodes]

# ------------------------------------------------------------------------------
# 3. Define the physical problem
# ------------------------------------------------------------------------------

print_section("Problem setup")

println("One system spin coupled to $nmodes bath spins with fused process-tensor memory.")
println()

@printf("time step dt           = %.4f\n", dt)
@printf("number of steps        = %d\n", nsteps)
@printf("final time             = %.4f\n", final_time)
@printf("bath modes             = %d\n", nmodes)
@printf("joint Hilbert dim      = %d\n", dsys * denv)

nmodes == 4 || @warn "This example is tuned for nmodes=4; using nmodes=$nmodes"

sys_phys = siteinds("S=1/2", 1)
env_phys = siteinds("S=1/2", nmodes)
env_liouv = liouv_sites(env_phys)

H_sys = OpSum()
H_sys += 1.0, "Sx", 1
system = spin_system(sys_phys, H_sys)

modes = SpinMode[]
for m in 1:nmodes
    rho_env_h = to_dm(MPS([env_phys[m]], ["Up"]))
    rho_env_l = to_liouville(rho_env_h; sites=[env_liouv[m]])
    H_mode = OpSum()
    H_mode += mode_w[m], "Sx", 1
    cpl_mode = OpSum()
    cpl_mode += mode_g[m], "Sz", 1, "Sz", 2
    push!(modes, spin_mode([env_liouv[m]], H_mode, rho_env_l; coupling=cpl_mode))
end
bath = spin_bath(modes)

rho_sys0_h = to_dm(MPS(sys_phys, ["Up"]))

joint_phys = Index[sys_phys[1], env_phys...]
joint_liouv = liouv_sites(joint_phys)

H_full = let
    H = OpSum()
    H += 1.0, "Sx", 1
    for m in 1:nmodes
        H += mode_w[m], "Sx", m + 1
        H += mode_g[m], "Sz", m + 1, "Sz", 1
    end
    H
end

joint_init = vcat(["Up"], fill("Up", nmodes))
psi_joint = MPS(joint_phys, joint_init)
rho_joint0_l = to_liouville(to_dm(psi_joint); sites=joint_liouv)

# ------------------------------------------------------------------------------
# 4. Main computation
# ------------------------------------------------------------------------------

print_section("Main computation")

println("Building multimode process tensor...")
pt = build_process_tensor(system, system.sites[1]; environment=bath, dt=dt, nsteps=nsteps)

println("Evolving reduced system with `evolve`...")
trajectory = evolve(pt, rho_sys0_h)

println("Computing joint ED reference at each snapshot...")
sx_ed, sy_ed, sz_ed = Float64[], Float64[], Float64[]
sx_pt, sy_pt, sz_pt = Float64[], Float64[], Float64[]
frob_err = Float64[]

ρ_pt0 = reduced_system_ρ(to_liouville(rho_sys0_h; sites=system.sites), dsys)
ρ_ed0 = partial_trace_system(to_hilbert(rho_joint0_l), dsys, denv)
obs_pt0 = pauli_expectations(ρ_pt0, σx, σy, σz)
obs_ed0 = pauli_expectations(ρ_ed0, σx, σy, σz)
push!(sx_pt, obs_pt0[1]); push!(sy_pt, obs_pt0[2]); push!(sz_pt, obs_pt0[3])
push!(sx_ed, obs_ed0[1]); push!(sy_ed, obs_ed0[2]); push!(sz_ed, obs_ed0[3])
push!(frob_err, norm(ρ_pt0 - ρ_ed0))

progress_stride = max(1, nsteps ÷ 20)

for k in 1:nsteps
    t = k * dt
    ρ_pt = reduced_system_ρ(trajectory.states_liouville[k], dsys)
    U_L = liouvillian_propagator_itensor(H_full, joint_liouv, t; alg=Exact())
    rho_joint_l = apply(U_L, copy(rho_joint0_l); cutoff=0.0, maxdim=typemax(Int))
    ρ_red = partial_trace_system(to_hilbert(rho_joint_l), dsys, denv)

    obs_pt = pauli_expectations(ρ_pt, σx, σy, σz)
    obs_ed = pauli_expectations(ρ_red, σx, σy, σz)
    push!(sx_pt, obs_pt[1])
    push!(sy_pt, obs_pt[2])
    push!(sz_pt, obs_pt[3])
    push!(sx_ed, obs_ed[1])
    push!(sy_ed, obs_ed[2])
    push!(sz_ed, obs_ed[3])
    push!(frob_err, norm(ρ_pt - ρ_red))

    if k == 1 || k == nsteps || k % progress_stride == 0
        update_status(@sprintf("  joint ED check  step %3d / %3d   t = %.3f", k, nsteps, t))
    end
end
finish_status("  joint ED reference complete.")

times = collect(0.0:dt:final_time)
@assert length(times) == length(sx_pt) == length(sx_ed)

# ------------------------------------------------------------------------------
# 5. Diagnostics and sanity checks
# ------------------------------------------------------------------------------

print_section("Diagnostics")

max_frob = maximum(frob_err)
@printf("max ‖ρ_PT − ρ_joint ED‖_F  = %.3e\n", max_frob)
@printf("⟨σ_x⟩ at t=0 (PT / ED)     = %.6f / %.6f\n", sx_pt[1], sx_ed[1])
@printf("⟨σ_y⟩ at t=0 (PT / ED)     = %.6f / %.6f\n", sy_pt[1], sy_ed[1])
@printf("⟨σ_x⟩ at t=T (PT / ED)     = %.6f / %.6f\n", sx_pt[end], sx_ed[end])

@assert all(isfinite, sx_pt) && all(isfinite, sx_ed)
@assert all(isfinite, frob_err)
@assert max_frob < joint_ed_frob_tol "‖ρ_PT − ρ_joint ED‖_F exceeds regression tolerance ($joint_ed_frob_tol)."

# ------------------------------------------------------------------------------
# 6. Plotting and saved outputs
# ------------------------------------------------------------------------------

print_section("Plotting")

lw = 2.4
fig = Figure(size=(880, 540))

ax1 = Axis(
    fig[1, 1];
    xlabel=L"t",
    title=latexstring(
        L"\mathrm{Reduced~system~observables},\ \Delta t = ",
        dt,
        L",\ N_{\mathrm{steps}} = ",
        nsteps,
        L",\ N_{\mathrm{modes}} = ",
        nmodes,
    ),
    ylabel=L"\langle \sigma_\alpha \rangle",
)
lines!(ax1, times, sx_ed; color=:steelblue, linestyle=:dash, linewidth=lw)
scatter!(ax1, times, sx_pt; marker=:star8, markersize=14, color=:steelblue, strokewidth=1.5, strokecolor=:white)
lines!(ax1, times, sz_ed; color=:firebrick, linestyle=:dash, linewidth=lw)
scatter!(ax1, times, sz_pt; marker=:circle, markersize=11, color=:firebrick, strokewidth=1.5, strokecolor=:white)
lines!(ax1, times, sy_ed; color=:darkgreen, linestyle=:dash, linewidth=lw)
scatter!(ax1, times, sy_pt; marker=:diamond, markersize=11, color=:darkgreen, strokewidth=1.5, strokecolor=:white)

axislegend(
    ax1,
    [
        [LineElement(color=:steelblue, linestyle=:dash, linewidth=lw)],
        [MarkerElement(marker=:star8, color=:steelblue, markersize=14, strokecolor=:white, strokewidth=1.5)],
        [LineElement(color=:firebrick, linestyle=:dash, linewidth=lw)],
        [MarkerElement(marker=:circle, color=:firebrick, markersize=11, strokecolor=:white, strokewidth=1.5)],
        [LineElement(color=:darkgreen, linestyle=:dash, linewidth=lw)],
        [MarkerElement(marker=:diamond, color=:darkgreen, markersize=11, strokecolor=:white, strokewidth=1.5)],
    ],
    [
        L"\langle \sigma_x \rangle\ (\mathrm{joint\ ED})",
        L"\langle \sigma_x \rangle\ (\mathrm{PT})",
        L"\langle \sigma_z \rangle\ (\mathrm{joint\ ED})",
        L"\langle \sigma_z \rangle\ (\mathrm{PT})",
        L"\langle \sigma_y \rangle\ (\mathrm{joint\ ED})",
        L"\langle \sigma_y \rangle\ (\mathrm{PT})",
    ];
    position=:rt,
    nbanks=2,
    fontsize=11,
    framevisible=true,
    backgroundcolor=(:white, 0.82),
    framewidth=1,
    rowgap=6,
    colgap=28,
    margin=(12, 12, 12, 12),
)

ax2 = Axis(
    fig[2, 1];
    xlabel=L"t",
    ylabel=L"\Vert \rho_{\mathrm{PT}} - \rho_{\mathrm{joint\ ED}} \Vert_{\mathrm{F}}",
    yscale=log10,
)
lines!(ax2, times, max.(frob_err, 1e-18); color=:black, linewidth=2)
scatter!(ax2, times, max.(frob_err, 1e-18); marker=:circle, markersize=11, color=:black, strokewidth=1.5, strokecolor=:white)
rowgap!(fig.layout, 18)
rowsize!(fig.layout, 2, Relative(0.28))

fig_path = joinpath(output_dir, "pt_tfim_multimode.png")
save(fig_path, fig)

println("Saved figure:")
println("  $fig_path")

# ------------------------------------------------------------------------------
# 7. Final summary
# ------------------------------------------------------------------------------

print_section("Summary")

println("Completed multimode process-tensor benchmark.")
println()
println("Main outputs:")
println("  figure: $fig_path")
println()
println("Main diagnostics:")
@printf("  max ‖ρ_PT − ρ_joint ED‖_F = %.3e\n", max_frob)
@printf("  final ⟨σ_x⟩ PT / ED       = %.6f / %.6f\n", sx_pt[end], sx_ed[end])
