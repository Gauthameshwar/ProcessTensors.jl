# Example: one system spin + one bath spin, TFIs-like Hamiltonian (unit terms). Process-tensor
# dynamics come from `evolve(pt, …)` (split bath+system step per `dt`, as in the library). The dashed
# reference is **continuous Schrödinger over each sub-interval** in the sense of one exact joint
# unitary per step: `ρ ↦ U ρ U†` with `U = exp(-im*dt*H_joint)` and full `H_joint = S^x_sys + S^x_env + S^z_sys S^z_env`
# on `[sys_phys, env_phys]`. 
#
# CairoMakie is not a dependency of the ProcessTensors package. This script provisions a tiny
# environment under `scripts/.plot_examples_env/` (gitignored) on first run: it `develop`s the
# repo and adds CairoMakie. From repo root:
#   julia scripts/example_pt_tfi_nm_ed_observables.jl

import Pkg

const REPO_ROOT = dirname(@__DIR__)
const _PLOT_ENV = joinpath(@__DIR__, ".plot_examples_env")

function _activate_plot_examples_env!()
    mkpath(_PLOT_ENV)
    Pkg.activate(_PLOT_ENV)
    manifest = joinpath(_PLOT_ENV, "Manifest.toml")
    if !isfile(manifest)
        Pkg.develop(Pkg.PackageSpec(path=REPO_ROOT))
        Pkg.add([Pkg.PackageSpec(name="CairoMakie")])
    else
        Pkg.instantiate()
    end
    return nothing
end

_activate_plot_examples_env!()

using CairoMakie
using ITensors
using LaTeXStrings
using LinearAlgebra
using ProcessTensors
include(joinpath(REPO_ROOT, "test", "time_evolution", "tebd_test_utils.jl"))

function _partial_trace_env(rho::AbstractMatrix{<:Number}, dsys::Int, denv::Int)
    rho4 = reshape(ComplexF64.(rho), dsys, denv, dsys, denv)
    out = zeros(ComplexF64, dsys, dsys)
    for e in 1:denv
        out .+= @view rho4[:, e, :, e]
    end
    return out
end

function _physical_sites_from_hilbert_mpo(rho::AbstractMPO{Hilbert})
    return [
        only(filter(i -> plev(i) == 0, inds(rho.core[j]))) for j in 1:length(rho.core)
    ]
end

"""⟨O⟩ = real(tr(ρ O)) for Hermitian O (ρ normalized)."""
function expectation_dense(rho::AbstractMatrix{<:Number}, O::AbstractMatrix{<:Number})
    return real(tr(rho * O))
end

function main(; dt::Float64 = 0.1, nsteps::Int = 24)
    sys_phys = siteinds("S=1/2", 1)
    env_phys = siteinds("S=1/2", 1)
    sys_liouv = liouv_sites(sys_phys)
    env_liouv = liouv_sites(env_phys)

    H_sys = OpSum()
    H_sys += 1.0, "Sx", 1
    system = spin_system(sys_phys, H_sys)

    rho_env0_h = to_dm(MPS(env_phys, ["Up"]))
    rho_env0_l = to_liouville(rho_env0_h; sites=env_liouv)
    H_env = OpSum()
    H_env += 1.0, "Sx", 1
    cpl = OpSum() + (1.0, "Sz", 1, "Sz", 2)
    mode = spin_mode(env_liouv, H_env, rho_env0_l; coupling=cpl)
    bath = spin_bath([mode])

    pt = build_process_tensor(system, system.sites[1]; environment=bath, dt=dt, nsteps=nsteps)

    rho_sys0_h = to_dm(MPS(sys_phys, ["Up"]))
    trajectory = evolve(pt, rho_sys0_h)

    joint_sites = [sys_phys[1], env_phys[1]]
    H_joint = OpSum()
    H_joint += 1.0, "Sx", 1
    H_joint += 1.0, "Sx", 2
    H_joint += 1.0, "Sz", 1, "Sz", 2
    H_joint_dense = dense_hamiltonian_matrix(H_joint, joint_sites)
    U_joint = exp(-1im * dt * ComplexF64.(Hermitian(H_joint_dense)))

    rho_joint = kron(
        hilbert_mpo_to_dense(rho_sys0_h, sys_phys),
        hilbert_mpo_to_dense(rho_env0_h, env_phys),
    )

    sx_pt = Float64[]
    sz_pt = Float64[]
    sy_pt = Float64[]
    sx_ed = Float64[]
    sz_ed = Float64[]
    sy_ed = Float64[]
    frob_err = Float64[]

    σx = ComplexF64[0 1; 1 0]
    σy = ComplexF64[0 -im; im 0]
    σz = ComplexF64[1 0; 0 -1]

    ρ_joint_ed = copy(rho_joint)
    for k in 0:(nsteps - 1)
        rho_pt_h = to_hilbert(trajectory.states_liouville[k + 1])
        ρ_pt = hilbert_mpo_to_dense(rho_pt_h, _physical_sites_from_hilbert_mpo(rho_pt_h))

        push!(sx_pt, expectation_dense(ρ_pt, σx))
        push!(sy_pt, expectation_dense(ρ_pt, σy))
        push!(sz_pt, expectation_dense(ρ_pt, σz))

        ρ_red = _partial_trace_env(ρ_joint_ed, 2, 2)
        push!(sx_ed, expectation_dense(ρ_red, σx))
        push!(sy_ed, expectation_dense(ρ_red, σy))
        push!(sz_ed, expectation_dense(ρ_red, σz))

        push!(frob_err, norm(ρ_pt - ρ_red))

        ρ_joint_ed = U_joint * ρ_joint_ed * U_joint'
    end

    max_err = maximum(frob_err)
    println("max Frobenius ‖ρ_PT − ρ_ref‖ (PT split-step vs joint U(Δt)=exp(-im*Δt*H_joint)) = ", max_err)

    times = trajectory.times

    ########  PLOT THE OBSERVABLES DYNAMICS  ########

    lw = 2.4
    fig = Figure(size=(880, 540))

    title = latexstring(
        L"\mathrm{Reduced~system~observables},\ \Delta t = ",
        dt,
        L",\ N_{\mathrm{steps}} = ",
        nsteps,
    )

    ax1 = Axis(
        fig[1, 1];
        xlabel=L"t",
        title=title,
        ylabel=L"\langle \sigma_\alpha \rangle",
    )

    lines!(ax1, times, sx_ed; color=:steelblue, linestyle=:dash, linewidth=lw)
    scatter!(ax1, times, sx_pt; marker=:star8, markersize=14, color=:steelblue, strokewidth=1.5, strokecolor=:white)

    lines!(ax1, times, sz_ed; color=:firebrick, linestyle=:dash, linewidth=lw)
    scatter!(ax1, times, sz_pt; marker=:circle, markersize=11, color=:firebrick, strokewidth=1.5, strokecolor=:white)

    lines!(ax1, times, sy_ed; color=:darkgreen, linestyle=:dash, linewidth=lw)
    scatter!(ax1, times, sy_pt; marker=:diamond, markersize=11, color=:darkgreen, strokewidth=1.5, strokecolor=:white)

    # nbanks=2 with ED triple then PT triple places ED column left, PT column right (same σ order).
    legend_elems = [
        [LineElement(color=:steelblue, linestyle=:dash, linewidth=lw)],
        [MarkerElement(marker=:star8, color=:steelblue, markersize=14, strokecolor=:white, strokewidth=1.5)],
        [LineElement(color=:firebrick, linestyle=:dash, linewidth=lw)],
        [MarkerElement(marker=:circle, color=:firebrick, markersize=11, strokecolor=:white, strokewidth=1.5)],
        [LineElement(color=:darkgreen, linestyle=:dash, linewidth=lw)],
        [MarkerElement(marker=:diamond, color=:darkgreen, markersize=11, strokecolor=:white, strokewidth=1.5)],
    ]
    legend_labels = [
        L"\langle \sigma_x \rangle\ (\mathrm{ED})",
        L"\langle \sigma_x \rangle\ (\mathrm{PT})",
        L"\langle \sigma_z \rangle\ (\mathrm{ED})",
        L"\langle \sigma_z \rangle\ (\mathrm{PT})",
        L"\langle \sigma_y \rangle\ (\mathrm{ED})",
        L"\langle \sigma_y \rangle\ (\mathrm{PT})",
    ]
    axislegend(
        ax1,
        legend_elems,
        legend_labels;
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
        ylabel=L"\Vert \rho_{\mathrm{PT}} - \rho_{\mathrm{ED}} \Vert_{\mathrm{F}}",
        yscale=log10,
    )
    lines!(ax2, times, max.(frob_err, 1e-18); color=:black, linewidth=2)
    scatter!(ax2, times, max.(frob_err, 1e-18); marker=:circle, markersize=11, color=:black, strokewidth=1.5, strokecolor=:white)
    rowgap!(fig.layout, 18)
    rowsize!(fig.layout, 2, Relative(0.28))

    outdir = joinpath(@__DIR__, "figures")
    mkpath(outdir)
    path = joinpath(outdir, "pt_tfi_nm_ed_observables.png")
    save(path, fig)
    println("Saved: ", path)
    return (; times, sx_pt, sz_pt, sx_ed, sz_ed, frob_err, fig_path=path)
end

main(;dt=0.1, nsteps=24)
