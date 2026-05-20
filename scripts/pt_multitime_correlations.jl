# Single-mode bath process tensor: two-time correlators ⟨A(t₂) B(t₁)⟩ heatmaps.
#
# Uses [`two_time_correlation_seq`](@ref) and sweeps the full (t₁, t₂) grid (not only t₂ ≥ t₁).
#
# From repo root:
#   julia scripts/pt_multitime_correlations.jl

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
using CairoMakie: Fixed
using ITensors, LinearAlgebra
using ProcessTensors

# Mathtext for axis labels / titles from `L"..."` strings.
set_theme!(
    Theme(
        fontsize=11,
        Axis=(
            titlefont=:bold,
            xlabelfont=:regular,
            ylabelfont=:regular,
        ),
    ),
)

"""
    _correlation_grid(system, bath, rho0_h, O_A, O_B, sys_phys; dt, n_times, ...)

Fill `grid[n₁+1, n₂+1] = ⟨A(t₂) B(t₁)⟩` for all `n₁, n₂ ∈ 0:(n_times-1)` using one
[`ProcessTensor`](@ref) with `nsteps = n_times + 1` (extra slot so same-time pairs at
`t = t_f` close with a terminal measurement) and [`two_time_correlation_seq`](@ref).
"""
function _correlation_grid(
    system,
    bath,
    rho0_h,
    O_A::OpSum,
    O_B::OpSum;
    dt::Real,
    n_times::Int,
)
    n_times >= 1 || throw(ArgumentError("n_times must be >= 1"))
    grid = fill(NaN + 0im * NaN, n_times, n_times)

    pt_nsteps = n_times + 1
    pt = build_process_tensor(
        system,
        system.sites[1];
        environment=bath,
        dt=dt,
        nsteps=pt_nsteps,
    )

    n_pairs = n_times * n_times
    k = 0
    for n1 in 0:(n_times - 1)
        for n2 in 0:(n_times - 1)
            k += 1
            seq = two_time_correlation_seq(
                pt,
                (O_A, n2),
                (O_B, n1);
                rho0=rho0_h,
            )
            val = evaluate_process(pt, seq)
            grid[n1 + 1, n2 + 1] = val
        end
        t1 = round(n1 * dt; digits=3)
        println("  finished row t₁ = $t1  ($(n1 + 1)/$(n_times), $k / $n_pairs pairs)")
    end
    return grid
end

function main(;
    dt::Float64=0.5,
    tf::Float64=5.5,
    rho_label::AbstractString="Dn",
)
    times = collect(0.0:dt:tf)
    n_times = length(times)
    println("=== PT single-mode: two-time correlation heatmaps (full t₁ × t₂ grid) ===")
    println("Δt = $dt, t_f = $tf, time points: ", join(string.(times), ", "))

    sys_phys = siteinds("S=1/2", 1)
    env_phys = siteinds("S=1/2", 1)
    env_liouv = liouv_sites(env_phys)

    # Single-site driven spin + bath (same model as scripts/pt_tfim_singlemode.jl).
    H_sys = OpSum()
    H_sys += 1.0, "Sx", 1
    system = spin_system(sys_phys, H_sys)

    rho_env0_h = to_dm(MPS(env_phys, ["Up"]))
    rho_env0_l = to_liouville(rho_env0_h; sites=env_liouv)
    H_env = OpSum()
    H_env += 1.0, "Sx", 1
    coupling = OpSum() + (1.0, "Sz", 1, "Sz", 2)
    mode = spin_mode(env_liouv, H_env, rho_env0_l; coupling=coupling)
    bath = spin_bath([mode])

    rho_sys0_h = to_dm(MPS(sys_phys, [rho_label]))

    # The operators are Pauli matrices, not the spin matrices. So we need to multiply by 2.
    O_Sz = OpSum()
    O_Sz += 2.0, "Sz", 1
    O_Sx = OpSum()
    O_Sx += 2.0, "Sx", 1
    O_Sy = OpSum()
    O_Sy += 2.0, "Sy", 1
    cases = [
        (L"\langle \sigma_z(t_2) \sigma_z(t_1) \rangle", O_Sz, O_Sz),
        (L"\langle \sigma_z(t_2) \sigma_x(t_1) \rangle", O_Sz, O_Sx),
        (L"\langle \sigma_x(t_2) \sigma_y(t_1) \rangle", O_Sx, O_Sy),
    ]

    grids = Vector{Matrix{ComplexF64}}(undef, length(cases))
    titles = Vector{LaTeXString}(undef, length(cases))

    ##############  COMPUTE CORRELATION GRIDS  ##############
    for (i, (title, O_A, O_B)) in enumerate(cases)
        println("\n[$i/$(length(cases))] ", title)
        titles[i] = title
        grids[i] = _correlation_grid(
            system,
            bath,
            rho_sys0_h,
            O_A,
            O_B;
            dt=dt,
            n_times=n_times,
        )
    end

    n = length(grids)
    t_lo, t_hi = times[1], times[end]
    absmax_re = maximum(
        maximum(abs, view(real(C), isfinite.(real(C)))) for C in grids;
        init=0.0,
    )
    absmax_im = maximum(
        maximum(abs, view(imag(C), isfinite.(imag(C)))) for C in grids;
        init=0.0,
    )
    absmax_re > 0 || (absmax_re = 1.0)
    absmax_im > 0 || (absmax_im = 1.0)

    ##############  PLOTTING  ##############
    outdir = joinpath(@__DIR__, "figures")
    mkpath(outdir)
    outfile = joinpath(outdir, "pt_multitime_correlations.png")

    panel = 250
    cb_w = 24.0
    ncols = n + 1
    nrows = 2
    fig = Figure(size=(n * panel + cb_w + 80, nrows * panel + 140), figure_padding=2)
    fig.layout.alignmode = Outside(12)

    panels = fig[2, 1] = GridLayout()
    panels.alignmode = Outside(2)
    colgap!(panels, 4)
    rowgap!(panels, 8)

    cmap = :balance
    row_specs = [
        (real.(grids), absmax_re, L"\mathrm{Re}\,\langle A(t_2) B(t_1) \rangle"),
        (imag.(grids), absmax_im, L"\mathrm{Im}\,\langle A(t_2) B(t_1) \rangle"),
    ]
    for (row, (grid_views, absmax, cb_label)) in enumerate(row_specs)
        for col in 1:n
            ax = Axis(
                panels[row, col];
                xlabel=row == nrows ? L"t_1" : "",
                ylabel=col == 1 ? L"t_2" : "",
                title=row == 1 ? titles[col] : "",
                aspect=DataAspect(),
                limits=(t_lo, t_hi, t_lo, t_hi),
                xticks=times,
                yticks=times,
            )
            heatmap!(
                ax,
                times,
                times,
                grid_views[col];
                colormap=cmap,
                colorrange=(-absmax, absmax),
                nan_color=(:white, 0.0),
            )
        end
        Colorbar(
            panels[row, ncols];
            colormap=cmap,
            limits=(-absmax, absmax),
            label=cb_label,
            height=panel,
            width=cb_w,
        )
        rowsize!(panels, row, Fixed(panel))
    end
    for col in 1:n
        colsize!(panels, col, Fixed(panel))
    end
    colsize!(panels, ncols, Fixed(cb_w))

    Label(
        fig[1, 1],
        latexstring(
            L"\mathrm{PT\ single-mode\ two-time\ correlations},\quad \Delta t = ",
            dt,
            L",\quad t_f = ",
            t_hi,
        );
        fontsize=13,
    )

    rowgap!(fig.layout, 2)
    colgap!(fig.layout, 2)
    resize_to_layout!(fig)
    save(outfile, fig; px_per_unit=2)

    println("\nSaved: ", outfile)
    println("✓ Done")
    return (; grids, times, outfile)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
