# Example: one system spin + one bath spin.
# Algorithm: split process-tensor `evolve(pt, ρ₀)`.
# Reference: exact joint `exp(t·L)` via `liouvillian_propagator_itensor(..., Exact())`, then partial trace.
#
# From repo root:
#   julia scripts/pt_tfim_singlemode.jl

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

using CairoMakie, LaTeXStrings
using ITensors, LinearAlgebra
using ITensors.Ops: Exact
using ProcessTensors

# -------- exact joint ED (Liouville propagator) --------------------------------

function _get_ed_observables(
    H_full,
    rho_joint0_l,
    joint_liouv,
    trajectory,
    dt,
    nsteps,
    dsys,
    denv,
    σx,
    σy,
    σz;
    label::String="ED",
)
    sx_ed, sy_ed, sz_ed = Float64[], Float64[], Float64[]
    frob_err = Float64[]
    t_start = time()
    println("[$label] start | nsteps=$nsteps dt=$dt joint_dim=$(dsys * denv)")

    for k in 0:(nsteps - 1)
        ρ_pt = _reduced_system_ρ(trajectory.states_liouville[k + 1], dsys)

        t = k * dt
        U_L = liouvillian_propagator_itensor(H_full, joint_liouv, t; exp_alg=Exact())
        rho_joint_l = apply(U_L, copy(rho_joint0_l); cutoff=0.0, maxdim=typemax(Int))
        ρ_red = _partial_trace_system(to_hilbert(rho_joint_l), dsys, denv)

        push!(sx_ed, real(tr(ρ_red * σx)))
        push!(sy_ed, real(tr(ρ_red * σy)))
        push!(sz_ed, real(tr(ρ_red * σz)))
        push!(frob_err, norm(ρ_pt - ρ_red))

        if k == 0 || k % 10 == 0 || k == nsteps - 1
            println(
                "[$label] step=$k/$nsteps t=$(round(t; digits=3)) ",
                "‖ρ_PT−ρ_ED‖_F=$(round(frob_err[end]; digits=4)) ",
                "elapsed=$(round(time() - t_start; digits=1))s",
            )
        end
    end

    println("[$label] done in $(round(time() - t_start; digits=1))s")
    return sx_ed, sy_ed, sz_ed, frob_err
end

function _reduced_system_ρ(state_l, dsys)
    rho_h = to_hilbert(state_l)
    sites = [
        only(filter(i -> plev(i) == 0 && hastags(i, "Site"), inds(rho_h.core[j])))
        for j in eachindex(rho_h.core)
    ]
    T = rho_h.core[1]
    for j in 2:length(rho_h.core)
        T *= rho_h.core[j]
    end
    A = Array(T, prime.(sites)..., sites...)
    return reshape(ComplexF64.(A), dsys, dsys)
end

function _partial_trace_system(rho_h, dsys, denv)
    sites = [
        only(filter(i -> plev(i) == 0 && hastags(i, "Site"), inds(rho_h.core[j])))
        for j in eachindex(rho_h.core)
    ]
    T = rho_h.core[1]
    for j in 2:length(rho_h.core)
        T *= rho_h.core[j]
    end
    A = Array(T, prime.(sites)..., sites...)
    ρ4 = reshape(ComplexF64.(A), dsys, denv, dsys, denv)
    ρ_red = zeros(ComplexF64, dsys, dsys)
    for e in 1:denv
        ρ_red .+= @view ρ4[:, e, :, e]
    end
    return ρ_red
end


function main(; dt::Float64=0.1, nsteps::Int=24)
    println("=== Process tensor (single bath mode): split PT vs joint Liouville ED ===")

    ##############  SETUP  ##############
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

    T_max = dt * nsteps
    println("Parameters: Δt=$dt, nsteps=$nsteps, T=$T_max, 1 system + 1 bath spin")

    println("Building process tensor...")
    pt = build_process_tensor(system, system.sites[1]; environment=bath, dt=dt, nsteps=nsteps)
    println("  PT built ($(nsteps) slabs, Δt=$dt)")

    rho_sys0_h = to_dm(MPS(sys_phys, ["Up"]))
    joint_phys = Index[sys_phys[1], env_phys[1]]
    joint_liouv = liouv_sites(joint_phys)

    H_full = OpSum()
    H_full += 1.0, "Sx", 1
    H_full += 1.0, "Sx", 2
    H_full += 1.0, "Sz", 1, "Sz", 2

    psi_joint = MPS(joint_phys, ["Up", "Up"])
    rho_joint0_l = to_liouville(to_dm(psi_joint); sites=joint_liouv)

    σx = ComplexF64[0 1; 1 0]
    σy = ComplexF64[0 -im; im 0]
    σz = ComplexF64[1 0; 0 -1]
    dsys, denv = 2, 2
    println("Joint Hilbert dimension: d_sys×d_env = $dsys×$denv = $(dsys * denv)")

    ##############  PROCESS TENSOR  ##############
    println("Process-tensor evolve (split schedule)...")
    t_pt = time()
    trajectory = evolve(pt, rho_sys0_h)
    println(
        "  done in $(round(time() - t_pt; digits=1))s, ",
        "$(length(trajectory.states_liouville)) snapshots",
    )

    ##############  EXACT JOINT ED  ##############
    sx_ed, sy_ed, sz_ed, frob_err = _get_ed_observables(
        H_full, rho_joint0_l, joint_liouv, trajectory, dt, nsteps, dsys, denv, σx, σy, σz;
        label="joint ED",
    )
    println(
        "max Frobenius ‖ρ_PT − ρ_joint ED‖ (Liouville Exact exp(t·L), split PT) = ",
        maximum(frob_err),
    )

    ##############  PT OBSERVABLES  ##############
    println("Extracting PT observables...")
    sx_pt, sy_pt, sz_pt = Float64[], Float64[], Float64[]
    for k in 0:(nsteps - 1)
        ρ_pt = _reduced_system_ρ(trajectory.states_liouville[k + 1], dsys)
        push!(sx_pt, real(tr(ρ_pt * σx)))
        push!(sy_pt, real(tr(ρ_pt * σy)))
        push!(sz_pt, real(tr(ρ_pt * σz)))
    end

    ##############  PLOTTING  ##############
    times = trajectory.times
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

    outdir = joinpath(@__DIR__, "figures")
    mkpath(outdir)
    path = joinpath(outdir, "pt_tfim_singlemode.png")
    save(path, fig)
    println("Saved: ", path)
    println(
        "  t=0: ⟨σ_x⟩_PT=$(round(sx_pt[1]; digits=4)), ⟨σ_x⟩_ED=$(round(sx_ed[1]; digits=4)); ",
        "t=$(round(last(times); digits=3)): ⟨σ_x⟩_PT=$(round(sx_pt[end]; digits=4)), ⟨σ_x⟩_ED=$(round(sx_ed[end]; digits=4))",
    )
    println("✓ Done")
    return (; times, sx_pt, sz_pt, sx_ed, sz_ed, frob_err, fig_path=path)
end

main(; dt=0.1, nsteps=24)
