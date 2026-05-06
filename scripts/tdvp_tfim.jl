# Liouville-space TDVP benchmark for the transverse-field Ising model (TFIM).
# Produces separate observable plots for the unitary and dissipative branches,
# plus dedicated conserved-quantity figures for each branch.
#
# Run from repo root:
#   julia --project=. scripts/tdvp_tfim.jl

using ProcessTensors, ITensors, LinearAlgebra
using CairoMakie, LaTeXStrings

const _ROOT ="figures",  @__DIR__
const _REPO_ROOT = joinpath(_ROOT, "figures", "..")

include(joinpath(_REPO_ROOT, "test", "time_evolution", "tebd_test_utils.jl"))

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

function exact_metrics(
    density_trajectory,
    H_dense,
    x_ops,
    z_ops,
)
    energy = Float64[]
    trace_err = Float64[]
    herm = Float64[]
    min_eig = Float64[]
    sx = Float64[]
    sz = Float64[]

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

function _run_tdvp_with_progress(
    label::String,
    ρ0_vec,
    operator,
    dt::Float64,
    nsteps::Int;
    nsite::Int,
    maxdim::Int,
    cutoff::Float64,
)
    states = Vector{typeof(ρ0_vec)}(undef, nsteps + 1)
    states[1] = copy(ρ0_vec)
    current = copy(ρ0_vec)
    t_start = time()
    println("[$label] start | nsteps=$nsteps dt=$dt nsite=$nsite maxdim=$maxdim cutoff=$cutoff")

    for step in 1:nsteps
        current = tdvp(
            operator,
            dt,
            current;
            time_step=dt,
            nsite=nsite,
            maxdim=maxdim,
            cutoff=cutoff,
            outputlevel=0,
        )
        states[step + 1] = current

        if step == 1 || step % 10 == 0 || step == nsteps
            elapsed = time() - t_start
            println("[$label] step=$step/$nsteps t=$(round(step * dt; digits=3)) bond=$(maxlinkdim(current)) elapsed=$(round(elapsed; digits=1))s")
        end
    end

    println("[$label] done in $(round(time() - t_start; digits=1))s")
    return states
end

function _run_dynamic_tdvp_with_progress(
    label::String,
    ρ0_vec,
    operator,
    dt::Float64,
    nsteps::Int;
    maxdim::Int,
    cutoff::Float64,
    plateau_patience::Int=2,
    switch_maxdim::Union{Nothing, Int}=nothing,
)
    states = Vector{typeof(ρ0_vec)}(undef, nsteps + 1)
    nsites = Vector{Int}(undef, nsteps)
    bond_dims = Vector{Int}(undef, nsteps + 1)
    states[1] = copy(ρ0_vec)
    current = copy(ρ0_vec)
    bond_dims[1] = maxlinkdim(current)
    current_nsite = 2
    stagnant_steps = 0
    t_start = time()

    println("[$label] start | nsteps=$nsteps dt=$dt dynamic(2→1) maxdim=$maxdim cutoff=$cutoff")
    for step in 1:nsteps
        nsites[step] = current_nsite
        prev_bond_dim = maxlinkdim(current)
        current = tdvp(
            operator,
            dt,
            current;
            time_step=dt,
            nsite=current_nsite,
            maxdim=maxdim,
            cutoff=cutoff,
            outputlevel=0,
        )
        states[step + 1] = current
        bond_dims[step + 1] = maxlinkdim(current)

        if current_nsite == 2
            if bond_dims[step + 1] <= prev_bond_dim
                stagnant_steps += 1
            else
                stagnant_steps = 0
            end

            reached_switch_cap = !isnothing(switch_maxdim) && bond_dims[step + 1] >= switch_maxdim
            if stagnant_steps >= plateau_patience || reached_switch_cap
                current_nsite = 1
                println("[$label] switch 2→1 at step=$step t=$(round(step * dt; digits=3)) bond_dim=$(bond_dims[step + 1])")
            end
        end

        if step == 1 || step % 10 == 0 || step == nsteps
            elapsed = time() - t_start
            println("[$label] step=$step/$nsteps t=$(round(step * dt; digits=3)) nsite=$(nsites[step]) bond=$(bond_dims[step + 1]) elapsed=$(round(elapsed; digits=1))s")
        end
    end
    println("[$label] done in $(round(time() - t_start; digits=1))s")
    return (; states, nsites, bond_dims)
end

function tdvp_methods_trajectory(ρ0_vec, operator, dt::Float64, T::Float64; maxdim::Int, cutoff::Float64)
    nsteps = round(Int, T / dt)
    times = collect(range(0.0, step=dt, length=nsteps + 1))
    one_states = _run_tdvp_with_progress("ONE", ρ0_vec, operator, dt, nsteps; nsite=1, maxdim=maxdim, cutoff=cutoff)
    two_states = _run_tdvp_with_progress("TWO", ρ0_vec, operator, dt, nsteps; nsite=2, maxdim=maxdim, cutoff=cutoff)
    dynamic = _run_dynamic_tdvp_with_progress("DYNAMIC", ρ0_vec, operator, dt, nsteps; maxdim=maxdim, cutoff=cutoff)
    return (
        times=times,
        one=(states=one_states, bond_dims=[maxlinkdim(state) for state in one_states]),
        two=(states=two_states, bond_dims=[maxlinkdim(state) for state in two_states]),
        dynamic=dynamic,
    )
end

function small_system_metrics(states, physical_sites, trace_bra, H_mpo, x_mpos, z_mpos)
    energy = Float64[]
    trace_err = Float64[]
    herm = Float64[]
    min_eig = Float64[]
    sx = Float64[]
    sz = Float64[]
    bond_dims = Int[]

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

const _METHOD_STYLES = Dict(
    :one => (; color=:dodgerblue3, linestyle=:dash, label="1TDVP"),
    :two => (; color=:darkorange2, linestyle=:solid, label="2TDVP"),
    :dynamic => (; color=:seagreen4, linestyle=:dot, label="Dynamic TDVP"),
)

function add_method_lines!(ax, times, method_data, field::Symbol)
    handles = AbstractPlot[]
    labels = String[]
    for method in (:one, :two, :dynamic)
        style = _METHOD_STYLES[method]
        curve = getfield(method_data[method], field)
        h = lines!(ax, times, curve; color=style.color, linestyle=style.linestyle, linewidth=2.6)
        push!(handles, h)
        push!(labels, style.label)
    end
    return handles, labels
end

function plot_observable(path, times, exact_times, exact_curve, method_data, field::Symbol; ylabel, title)
    fig = Figure(size=(900, 520))
    ax = Axis(fig[1, 1]; xlabel=L"$t$", ylabel=ylabel, title=title)
    handles = AbstractPlot[]
    labels = String[]

    h_exact = lines!(ax, exact_times, exact_curve; color=:black, linewidth=4.8)
    push!(handles, h_exact)
    push!(labels, "ED")

    method_handles, method_labels = add_method_lines!(ax, times, method_data, field)
    append!(handles, method_handles)
    append!(labels, method_labels)
    Legend(fig[2, 1], handles, labels; orientation=:horizontal, nbanks=1, tellwidth=false, tellheight=true)
    save(path, fig)
    println("Saved: ", path)
end

function plot_unitary_conserved(path, times, exact_times, exact_data, method_data; title)
    fig = Figure(size=(1200, 700))
    ax_energy = Axis(fig[1, 1]; xlabel=L"$t$", ylabel=L"$\langle H \rangle (t)$", title=title)
    ax_drift = Axis(fig[1, 2]; xlabel=L"$t$", ylabel="energy drift")
    ax_trace = Axis(fig[1, 3]; xlabel=L"$t$", ylabel="trace error", yscale=log10)
    ax_herm = Axis(fig[2, 1]; xlabel=L"$t$", ylabel="Hermiticity defect", yscale=log10)
    ax_psd = Axis(fig[2, 2]; xlabel=L"$t$", ylabel="min eig")
    ax_bond = Axis(fig[2, 3]; xlabel=L"$t$", ylabel="max bond dim")

    exact_energy = lines!(ax_energy, exact_times, exact_data.energy; color=:black, linewidth=4.8)
    exact_drift = lines!(ax_drift, exact_times, exact_data.energy_drift; color=:black, linewidth=4.8)
    exact_trace = lines!(ax_trace, exact_times, max.(exact_data.trace_err, eps(Float64)); color=:black, linewidth=4.8)
    exact_herm = lines!(ax_herm, exact_times, max.(exact_data.hermiticity, eps(Float64)); color=:black, linewidth=4.8)
    exact_psd = lines!(ax_psd, exact_times, exact_data.min_eig; color=:black, linewidth=4.8)

    handles = AbstractPlot[exact_energy]
    labels = ["ED"]

    for method in (:one, :two, :dynamic)
        style = _METHOD_STYLES[method]
        data = method_data[method]
        h = lines!(ax_energy, times, data.energy; color=style.color, linestyle=style.linestyle, linewidth=2.4)
        lines!(ax_drift, times, max.(data.energy_drift, eps(Float64)); color=style.color, linestyle=style.linestyle, linewidth=2.4)
        lines!(ax_trace, times, max.(data.trace_err, eps(Float64)); color=style.color, linestyle=style.linestyle, linewidth=2.4)
        lines!(ax_herm, times, max.(data.hermiticity, eps(Float64)); color=style.color, linestyle=style.linestyle, linewidth=2.4)
        lines!(ax_psd, times, data.min_eig; color=style.color, linestyle=style.linestyle, linewidth=2.4)
        lines!(ax_bond, times, data.bond_dims; color=style.color, linestyle=style.linestyle, linewidth=2.4)
        push!(handles, h)
        push!(labels, style.label)
    end

    Legend(fig[3, 1:3], handles, labels; orientation=:horizontal, nbanks=1, tellwidth=false, tellheight=true)
    save(path, fig)
    println("Saved: ", path)
end

function plot_dissipative_conserved(path, times, exact_times, exact_data, method_data; title)
    fig = Figure(size=(1100, 700))
    ax_trace = Axis(fig[1, 1]; xlabel=L"$t$", ylabel="trace error", title=title, yscale=log10)
    ax_herm = Axis(fig[1, 2]; xlabel=L"$t$", ylabel="Hermiticity defect", yscale=log10)
    ax_psd = Axis(fig[2, 1]; xlabel=L"$t$", ylabel="min eig")
    ax_bond = Axis(fig[2, 2]; xlabel=L"$t$", ylabel="max bond dim")

    exact_trace = lines!(ax_trace, exact_times, max.(exact_data.trace_err, eps(Float64)); color=:black, linewidth=4.8)
    lines!(ax_herm, exact_times, max.(exact_data.hermiticity, eps(Float64)); color=:black, linewidth=4.8)
    lines!(ax_psd, exact_times, exact_data.min_eig; color=:black, linewidth=4.8)

    handles = AbstractPlot[exact_trace]
    labels = ["ED"]

    for method in (:one, :two, :dynamic)
        style = _METHOD_STYLES[method]
        data = method_data[method]
        h = lines!(ax_trace, times, max.(data.trace_err, eps(Float64)); color=style.color, linestyle=style.linestyle, linewidth=2.4)
        lines!(ax_herm, times, max.(data.hermiticity, eps(Float64)); color=style.color, linestyle=style.linestyle, linewidth=2.4)
        lines!(ax_psd, times, data.min_eig; color=style.color, linestyle=style.linestyle, linewidth=2.4)
        lines!(ax_bond, times, data.bond_dims; color=style.color, linestyle=style.linestyle, linewidth=2.4)
        push!(handles, h)
        push!(labels, style.label)
    end

    Legend(fig[3, 1:2], handles, labels; orientation=:horizontal, nbanks=1, tellwidth=false, tellheight=true)
    save(path, fig)
    println("Saved: ", path)
end

function main()
    println("=== Liouville-space TDVP benchmark: TFIM unitary + dissipative ===")

    N = 4
    T_unitary = 4.0
    T_diss = 9.0
    dt = 0.05
    maxdim = 128
    cutoff = 1e-10

    physical_sites = siteinds("S=1/2", N)
    liouv_sites_shared = liouv_sites(physical_sites)
    trace_bra = vectorized_identity_state(physical_sites, liouv_sites_shared)
    os_H = tfim_hamiltonian(N; J=1.0, h=1.2)
    H_mpo = MPO(os_H, physical_sites)
    H_dense = dense_hamiltonian_matrix(os_H, physical_sites)
    jump_ops = tfim_decay_jump_ops(N; γ=0.5)

    ψ0 = MPS(physical_sites, fill("Up", N))
    ρ0 = to_dm(ψ0)
    ρ0_vec = to_liouville(ρ0; sites=liouv_sites_shared)

    x_ops = [dense_one_site_operator("X", physical_sites, j) for j in 1:N]
    z_ops = [dense_one_site_operator("Z", physical_sites, j) for j in 1:N]
    x_mpos = single_site_pauli_mpos("X", physical_sites)
    z_mpos = single_site_pauli_mpos("Z", physical_sites)

    d = prod(dim.(physical_sites))
    vec0 = vec(ComplexF64.(hilbert_mpo_to_dense(ρ0, physical_sites)))

    L_unitary_mpo = MPO_Liouville(os_H, liouv_sites_shared; jump_ops=Tuple{Number, String, Int}[])
    L_diss_mpo = MPO_Liouville(os_H, liouv_sites_shared; jump_ops=jump_ops)
    L_unitary_dense = dense_liouvillian_matrix(os_H, Tuple{Number, String, Int}[], physical_sites, liouv_sites_shared)
    L_diss_dense = dense_liouvillian_matrix(os_H, jump_ops, physical_sites, liouv_sites_shared)

    unitary_runs = tdvp_methods_trajectory(ρ0_vec, L_unitary_mpo, dt, T_unitary; maxdim=maxdim, cutoff=cutoff)
    dissipative_runs = tdvp_methods_trajectory(ρ0_vec, L_diss_mpo, dt, T_diss; maxdim=maxdim, cutoff=cutoff)

    unitary_exact_densities = exact_density_trajectory(L_unitary_dense, vec0, d, unitary_runs.times)
    dissipative_exact_densities = exact_density_trajectory(L_diss_dense, vec0, d, dissipative_runs.times)
    unitary_exact = exact_metrics(unitary_exact_densities, H_dense, x_ops, z_ops)
    dissipative_exact = exact_metrics(dissipative_exact_densities, H_dense, x_ops, z_ops)

    unitary_method_data = Dict(
        :one => small_system_metrics(unitary_runs.one.states, physical_sites, trace_bra, H_mpo, x_mpos, z_mpos),
        :two => small_system_metrics(unitary_runs.two.states, physical_sites, trace_bra, H_mpo, x_mpos, z_mpos),
        :dynamic => small_system_metrics(unitary_runs.dynamic.states, physical_sites, trace_bra, H_mpo, x_mpos, z_mpos),
    )
    dissipative_method_data = Dict(
        :one => small_system_metrics(dissipative_runs.one.states, physical_sites, trace_bra, H_mpo, x_mpos, z_mpos),
        :two => small_system_metrics(dissipative_runs.two.states, physical_sites, trace_bra, H_mpo, x_mpos, z_mpos),
        :dynamic => small_system_metrics(dissipative_runs.dynamic.states, physical_sites, trace_bra, H_mpo, x_mpos, z_mpos),
    )

    plot_observable(
        joinpath(_ROOT, "figures", "tdvp_tfim_unitary_mx.png"),
        unitary_runs.times,
        unitary_runs.times,
        unitary_exact.sx,
        unitary_method_data,
        :sx;
        ylabel=L"$\langle \overline{\sigma}_x \rangle (t)$",
        title="Unitary TFIM TDVP (N=$N, dt=$dt)",
    )
    plot_observable(
        joinpath(_ROOT, "figures", "tdvp_tfim_unitary_mz.png"),
        unitary_runs.times,
        unitary_runs.times,
        unitary_exact.sz,
        unitary_method_data,
        :sz;
        ylabel=L"$\langle \overline{\sigma}_z \rangle (t)$",
        title="Unitary TFIM TDVP (N=$N, dt=$dt)",
    )
    plot_unitary_conserved(
        joinpath(_ROOT, "figures", "tdvp_tfim_unitary_conserved.png"),
        unitary_runs.times,
        unitary_runs.times,
        unitary_exact,
        unitary_method_data;
        title="Unitary TFIM conserved quantities",
    )

    plot_observable(
        joinpath(_ROOT, "figures", "tdvp_tfim_dissipative_mx.png"),
        dissipative_runs.times,
        dissipative_runs.times,
        dissipative_exact.sx,
        dissipative_method_data,
        :sx;
        ylabel=L"$\langle \overline{\sigma}_x \rangle (t)$",
        title="Dissipative TFIM TDVP (N=$N, dt=$dt)",
    )
    plot_observable(
        joinpath(_ROOT, "figures", "tdvp_tfim_dissipative_mz.png"),
        dissipative_runs.times,
        dissipative_runs.times,
        dissipative_exact.sz,
        dissipative_method_data,
        :sz;
        ylabel=L"$\langle \overline{\sigma}_z \rangle (t)$",
        title="Dissipative TFIM TDVP (N=$N, dt=$dt)",
    )
    plot_dissipative_conserved(
        joinpath(_ROOT, "figures", "tdvp_tfim_dissipative_conserved.png"),
        dissipative_runs.times,
        dissipative_runs.times,
        dissipative_exact,
        dissipative_method_data;
        title="Dissipative TFIM physicality diagnostics",
    )

    println("Dynamic TDVP unitary nsite schedule: ", unitary_runs.dynamic.nsites)
    println("Dynamic TDVP dissipative nsite schedule: ", dissipative_runs.dynamic.nsites)
    println("✓ Done")
end

main()
