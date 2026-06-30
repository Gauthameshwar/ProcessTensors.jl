# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors #src
# SPDX-License-Identifier: MIT #src
# #src
# File: docs/literate/tutorials/05_process_tensor_singlemode.jl #src
# Contributor: Gauthameshwar S. #src
# #src
# Literate tutorial source: building and using a single-mode process tensor. #src

# # Single-Mode Process Tensor
#
# The previous tutorials described density matrices, Liouville-space dynamics,
# unitary evolution, and dissipative Lindblad evolution.
#
# Those descriptions are still time-local. At each time step, we apply a fixed
# map to the current density matrix:
#
# ```math
# |\rho(t+\Delta t)\rangle\rangle
# =
# \mathcal{E}_{\Delta t}
# |\rho(t)\rangle\rangle.
# ```
#
# A process tensor is more general. It stores how a system responds to a whole
# sequence of interventions over time.
#
# Instead of asking only for a state trajectory, we can ask:
#
# - What happens if I prepare this state at the beginning?
# - What if I insert an observable at an intermediate time?
# - What if I leave one output open and inspect the reduced state?
# - What if I postselect on an outcome?
# - What if I use the same environment influence with different instruments?
#
# In `ProcessTensors.jl`, a process tensor is represented as a Liouville-space
# MPO over time legs.
#
# ```text
# time 0        time 1        time 2
#
#  input        input         input
#    |            |             |
#  [ PT ] —— memory —— [ PT ] —— memory —— [ PT ]
#    |            |             |
# output       output        output
# ```
#
# The memory links carry the influence of the environment from one time step to
# the next. Instruments are then contracted onto the input/output legs.

# ## Setup
#
using ITensors
import LinearAlgebra
using ProcessTensors

roundreal(x; digits=8) = round(real(x); digits=digits)

# ## Ingredients to build a process tensor
#
# A process tensor needs four physical ingredients:
#
# - a system,
# - an environment,
# - a system-environment coupling,
# - a time grid.
#
# In this tutorial, the system is one spin:
#
# ```math
# H_S = h S^x_S.
# ```
#
# The environment is one spin bath mode:
#
# ```math
# H_B = \omega_B S^z_B.
# ```
#
# The system and bath interact through
#
# ```math
# H_{SB} = g S^z_B S^z_S.
# ```
#
# This is the smallest nontrivial setting where the process tensor has a memory
# link. The environment is finite, so it can carry information from one time to
# another.

# ### System
#
# We build the system from ordinary Hilbert-space spin sites. The constructor
# `spin_system` converts them internally to canonical Liouville sites.

system_sites = siteinds("S=1/2", 1)

h = 0.7

H_S = OpSum()
H_S += h, "Sx", 1

system = spin_system(system_sites, H_S)

println("System:")
println(system)

@assert system isa SpinSystem
@assert length(system.sites) == 1
@assert dim(only(system.sites)) == 4

# The initial system state is
#
# ```math
# |+\rangle =
# \frac{|\uparrow\rangle + |\downarrow\rangle}{\sqrt{2}}.
# ```

ψS0 = MPS(ComplexF64[1 / sqrt(2), 1 / sqrt(2)], system_sites)
ρS0 = to_dm(ψS0)

@assert ρS0 isa MPO{Hilbert}
@assert abs(tr(ρS0) - 1) < 1e-12

# ### Bath mode
#
# The bath mode is also a spin. Bath modes are stored in Liouville space because
# the process tensor is built from density matrices and superoperators.

bath_sites = siteinds("S=1/2", 1)
bath_sites_L = liouv_sites(bath_sites)

ωB = 1.1

H_B = OpSum()
H_B += (ωB / 2), "Sz", 1

ψB0 = MPS(bath_sites, ["Dn"])
ρB0 = to_dm(ψB0)
ρB0_L = to_liouville(ρB0; sites=bath_sites_L)

@assert ρB0_L isa MPS{Liouville}

# The coupling OpSum is written in a local two-site convention:
#
# - site `1` is the bath mode,
# - site `2` is the system coupling site.
#
# This local convention lets the same coupling be inserted into the joint
# bath-system Liouville propagator.

g = 0.35

H_SB = OpSum()
H_SB += g, "Sz", 1, "Sz", 2

mode = spin_mode(
    bath_sites_L,
    H_B,
    ρB0_L;
    coupling=H_SB,
)

environment = spin_bath([mode])

println("Environment:")
println(environment)

@assert mode isa SpinMode
@assert environment isa SpinBath

# !!! tip "Nearby variation: use a bosonic mode"
#     The spin bath can be changed into a bosonic bath by changing only the mode
#     constructor and the site family.
#
#     ```julia
#     boson_sites = siteinds("Boson", 1; dim=4)
#     boson_sites_L = liouv_sites(boson_sites)
#
#     H_boson = OpSum()
#     H_boson += Ω, "N", 1
#
#     ψb0 = MPS(boson_sites, ["0"])
#     ρb0_L = to_liouville(to_dm(ψb0); sites=boson_sites_L)
#
#     coupling_boson = OpSum()
#     coupling_boson += g, "N", 1, "Sz", 2
#
#     mode_b = bosonic_mode(
#         boson_sites_L,
#         H_boson,
#         ρb0_L;
#         n_max=3,
#         coupling=coupling_boson,
#     )
#
#     environment_b = bosonic_bath([mode_b])
#     ```
#
#     The process-tensor construction call is unchanged. Only the bath model has
#     changed.

# ## [Building the process tensor](@id building-the-process-tensor)
#
# The time grid is set by `dt` and `nsteps`.
#
# The call
#
# ```julia
# pt = build_process_tensor(system; environment, dt, nsteps)
# ```
#
# builds one process-tensor core per time step.
#
# By default, the system's own one-step Liouville propagation is embedded into
# the cores. This means the default instrument between time steps is the identity
# operation, not an extra system propagator.

dt = 0.05
nsteps = 5

pt = build_process_tensor(
    system;
    environment=environment,
    dt=dt,
    nsteps=nsteps,
    alg=Exact(),
    sys_alg=Trotter{2}(),
)

println("Process tensor with one spin bath mode:")
println(pt)

@assert pt isa ProcessTensor
@assert pt.nsteps == nsteps
@assert pt.dt == dt
@assert pt.environment isa SpinBath
@assert pt.embed_system_propagation

# It is useful to compare this with the no-environment baseline.
#
# With `environment=nothing`, the object stores the time-local system
# propagation without an explicit bath memory link.

pt_markov = build_process_tensor(
    system;
    dt=dt,
    nsteps=nsteps,
    sys_alg=Trotter{2}(),
)

println("No-environment baseline:")
println(pt_markov)

println("maxlinkdim(pt_markov) = ", maxlinkdim(pt_markov))
println("maxlinkdim(pt)        = ", maxlinkdim(pt))

@assert maxlinkdim(pt) >= maxlinkdim(pt_markov)

# In the Markovian limit, the process tensor decomposes to a product MPO-like object in time. 
# Since there is no memory link between interventions, the process becomes a product of the 
# same system unitary at each time step. 
# 
# ### Time legs
#
# At every process-tensor time label `k`, there is an input leg and an output
# leg.
#
# In this package:
#
# - input legs have prime level `1`,
# - output legs have prime level `0`,
# - both carry a `tstep=k` tag.
# - both have the same `tags`, `ID`, and liouville `dim`. 
#
# These helpers let us inspect the legs without manually searching through
# ITensor indices.

println("Input leg at time 0:")
println(input_sites(pt, 0))

println("Output leg at time 0:")
println(output_sites(pt, 0))

println("Legs connected by the evolve slot at step 1:")
println(coupling_times(pt, 1))

@assert plev(only(input_sites(pt, 0))) == 1
@assert plev(only(output_sites(pt, 0))) == 0

# The `coupling_times` returns an output index of `tstep=0` and an input index of `tstep=1`.
# This is because while propagating the initial state in a process tensor, we contract the 
# input leg of `tstep=0` with the output leg of `tstep=1` of the previous time step. If there is 
# no intervention at this timestep, and we let the system do its unitary evolution, we embed an 
# `IdentityOperation` at this timestep.
# 
# !!! warning "Reuse Liouville indices"
#     Process-tensor contractions depend on exact ITensor index identity. Use
#     the sites stored by the system, bath modes, and process tensor instead of
#     recreating visually similar indices.

# ## [Evolving reduced states](@id evolving-reduced-states)
#
# The simplest way to use a process tensor is to ask for the reduced system
# states generated from an initial density matrix.
#
# The high-level call is:
#
# ```julia
# trajectory = evolve(pt, ρS0)
# ```
#
# It returns:
#
# - `trajectory.times`,
# - `trajectory.states_liouville`,
# - `trajectory.states_hilbert`.
#
# The Hilbert states are density-matrix MPOs reconstructed from the
# Liouville-space outputs.

trajectory = evolve(pt, ρS0)

println("Sample times:")
println(trajectory.times)

println("Number of returned states:")
println(length(trajectory.states_hilbert))

@assert length(trajectory.times) == nsteps
@assert length(trajectory.states_hilbert) == nsteps
@assert all(ρ -> abs(tr(ρ) - 1) < 1e-8, trajectory.states_hilbert)

# Let us measure one simple observable along the trajectory:
#
# ```math
# \langle S^z(t)\rangle =
# \operatorname{Tr}[S^z\rho(t)]
# = \langle\langle S^z | \rho(t) \rangle\rangle.
# ```
#
# We vectorize both $\rho(t)$ and $S^z$ on the **same** Liouville sites, then use
# the two-argument overlap `inner(Sz_L, ρL)` from [Liouville-Space Basics](@ref).

Sz = OpSum()
Sz += 1.0, "Sz", 1

Sz_mpo = MPO(Sz, system_sites)
system_sites_L = liouv_sites(system_sites)
Sz_L = to_liouville(Sz_mpo; sites=system_sites_L)

mz = [
    begin
        ρL = to_liouville(ρ; sites=system_sites_L)
        real(inner(Sz_L, ρL))
    end
    for ρ in trajectory.states_hilbert
]

println("⟨Sz⟩ along the process-tensor trajectory:")
println(roundreal.(mz))

@assert all(isfinite, mz)

# Now compare with the no-environment baseline. The code is identical; only the
# process tensor changes.

trajectory_markov = evolve(pt_markov, ρS0)

mz_markov = [
    begin
        ρL = to_liouville(ρ; sites=system_sites_L)
        real(inner(Sz_L, ρL))
    end
    for ρ in trajectory_markov.states_hilbert
]

println("Final ⟨Sz⟩ without explicit bath = ", roundreal(last(mz_markov)))
println("Final ⟨Sz⟩ with spin bath        = ", roundreal(last(mz)))

# The two results need not agree. The whole point of the process tensor is that
# the bath can carry memory between time steps.

@assert isfinite(last(mz_markov))
@assert isfinite(last(mz))

# ## [Instrument schedules](@id instrument-schedules)
#
# A process tensor becomes useful when we contract it with instruments.
#
# An instrument says what we do at a given time leg:
#
# - prepare a state,
# - insert an observable,
# - connect one time to the next,
# - trace out a leg,
# - measure an outcome, or
# - leave an output open.
#
# The default schedule for a process tensor with embedded propagation uses
# `IdentityOperation()` between time steps.

seq = default_schedule(pt)

println("Default schedule:")
println(seq)

@assert seq isa InstrumentSeq

# ### Normalization as a fully closed process
#
# If every leg is closed, `evaluate_process` returns a scalar.
#
# The schedule below prepares `ρS0` at the beginning and traces the final output.
# Physically, this asks for the total probability of the process.

seq_norm = default_schedule(pt)
add!(seq_norm, StatePreparation(ρS0), 0)
add!(seq_norm, TraceOut(), pt.nsteps)

norm_val = evaluate_process(pt, seq_norm)

println("Closed process value:")
println(norm_val)

@assert norm_val isa ComplexF64
@assert abs(norm_val - 1) < 1e-8

# ### Leaving an output open
#
# The same schedule, if leaves the final output leg open, returns the final reduced system state.
#
# The instrument `OpenOutput()` traces the current input leg and keeps the
# previous output leg open. This is useful when we want a reduced state at an
# intermediate cut.

seq_open = default_schedule(pt)
add!(seq_open, StatePreparation(ρS0), 0)
add!(seq_open, OpenOutput(), 2)
add!(seq_open, TraceOut(), pt.nsteps)

open_result = evaluate_process(pt, seq_open)

println("Open-output result type:")
println(typeof(open_result))

@assert open_result isa MPO{Liouville}

# ### Final expectation value
#
# To compute a final observable, replace the final trace with an observable
# insertion.
#
# This gives
#
# ```math
# \operatorname{Tr}[S^z\rho(t_{\mathrm{final}})]
# = \langle\langle \rho(t_{\mathrm{final}}) | S^z \rangle\rangle.
# ```

seq_final_sz = default_schedule(pt)
add!(seq_final_sz, StatePreparation(ρS0), 0)
add!(seq_final_sz, ObservableMeasurement(Sz), pt.nsteps)

final_sz_from_schedule = evaluate_process(pt, seq_final_sz)

println("Final ⟨Sz⟩ from schedule:")
println(final_sz_from_schedule)

println("Final ⟨Sz⟩ from evolve:")
println(last(mz))

@assert abs(real(final_sz_from_schedule) - last(mz)) < 1e-8

# For ordinary state trajectories, prefer `evolve(pt, ρ0)`. `OpenOutput` is the
# lower-level schedule ingredient that makes such state extraction possible.

# ### Lazy instruments and dense instruments
#
# Every schedule above used instruments **lazily**: `add!` stores what we want
# to do, and the package materializes the corresponding ITensor only when the
# process tensor is contracted — much like an `OpSum` before `MPO(...)`.
#
# Sometimes we want to inspect or define the dense map ourselves. The package
# supports both paths:
#
# - **lazy path:** high-level instruments such as `ObservableMeasurement`,
#   `IdentityOperation`, `OpenOutput`, and `left_right_operator`,
# - **dense path:** materialize with `instrument_itensor`, or wrap a custom map
#   in `CustomTwoLegInstrument`.
#
# The same trace-reducing postselection illustrates both paths.
#
# A spin-up projector is
#
# ```math
# P_\uparrow =
# |\uparrow\rangle\langle\uparrow|
# =
# \frac{1}{2}I + S^z,
# ```
#
# and the corresponding map is $\rho \mapsto P_\uparrow \rho P_\uparrow$. This
# map is not trace preserving; the closed scalar from `evaluate_process` is the
# probability that postselection succeeds.

Pup = OpSum()
Pup += 0.5, "Id", 1
Pup += 1.0, "Sz", 1

Pup_mpo = MPO(Pup, system_sites)

lazy_filter = left_right_operator(Pup_mpo, Pup_mpo)

seq_filter_lazy = default_schedule(pt)
add!(seq_filter_lazy, StatePreparation(ρS0), 0)
add!(seq_filter_lazy, lazy_filter, 2)
add!(seq_filter_lazy, TraceOut(), pt.nsteps)

prob_filter_lazy = evaluate_process(pt, seq_filter_lazy)

println("Postselected trace from lazy instrument:")
println(prob_filter_lazy)

@assert 0 <= real(prob_filter_lazy) <= 1 + 1e-10

# Materialize the same instrument on the concrete process-tensor legs.
# `coupling_times(pt, step)` returns `(out_prev, in_curr)`: the output leg at
# `step - 1` and the input leg at `step`.

out_prev, in_curr = coupling_times(pt, 2)

dense_filter_tensor = instrument_itensor(
    lazy_filter,
    in_curr,
    out_prev,
    2,
)

println("Dense filter instrument:")
println(dense_filter_tensor)

dense_filter = CustomTwoLegInstrument(
    dense_filter_tensor,
    in_curr,
    out_prev,
)

seq_filter_dense = default_schedule(pt)
add!(seq_filter_dense, StatePreparation(ρS0), 0)
add!(seq_filter_dense, dense_filter, 2)
add!(seq_filter_dense, TraceOut(), pt.nsteps)

prob_filter_dense = evaluate_process(pt, seq_filter_dense)

println("Postselected trace from dense custom instrument:")
println(prob_filter_dense)

@assert abs(prob_filter_dense - prob_filter_lazy) < 1e-10

# We can also contract the materialized tensors by hand. Most users should prefer
# `evaluate_process`; the loop below shows what the high-level call is doing.

prob_filter_manual = let
    dense_instruments = create_instruments(pt, seq_filter_dense)
    manual_result = pt.core[1] * dense_instruments[1]
    for step in 1:(pt.nsteps - 1)
        manual_result *= dense_instruments[step + 1]
        manual_result *= pt.core[step + 1]
    end
    final_instr = resolve_instrument(
        seq_filter_dense,
        pt.nsteps,
        seq_filter_dense.default,
    )
    final_out, _ = coupling_times(pt, pt.nsteps)
    manual_result *= instrument_itensor(
        final_instr,
        final_out,
        pt.nsteps - 1,
    )
    ComplexF64(scalar(manual_result))
end

println("Postselected trace from manual dense contraction:")
println(prob_filter_manual)

@assert abs(prob_filter_manual - prob_filter_lazy) < 1e-10

# !!! note "Lazy versus dense instruments"
#     The lazy path is the recommended interface:
#
#     ```julia
#     add!(seq, left_right_operator(Pup_mpo, Pup_mpo), 2)
#     ```
#
#     The dense path is useful when you want to inspect the ITensor, define a
#     custom operation, or debug a contraction:
#
#     ```julia
#     dense = instrument_itensor(instr, in_curr, out_prev, step)
#     custom = CustomTwoLegInstrument(dense, in_curr, out_prev)
#     ```
#
#     Both paths describe the same physical intervention when the dense tensor is
#     materialized from the same lazy instrument.

# ### Collapse and reprepare at one time step
#
# A causal intervention can filter the state leaving time step `k-1` and set the
# state entering time step `k`. For a spin-up measurement followed by repreparing
# $|\!\uparrow\rangle$, use a [`ProductInstrument`](@ref):
#
# - [`observable_measurement`](@ref) on the **output** leg (`plev=0`),
# - [`state_preparation`](@ref) on the **input** leg (`plev=1`, the default).
#
# The product order does not matter as long as the two factors sit on opposite
# legs.
#
# This is a completely positive map. It is **not** trace preserving when the
# measurement is probabilistic: the closed scalar from `evaluate_process` is the
# probability of the outcome, not `1`.
#
# This post-selection step says we measure the system in the spin-up state, 
# and then prepare the system in the spin-up state after realising that outcome.

Pup = OpSum()
Pup += 0.5, "Id", 1
Pup += 1.0, "Sz", 1

ρ_up = to_dm(MPS(system_sites, ["Up"]))

seq_filter = default_schedule(pt)
add!(seq_filter, StatePreparation(ρS0), 0)
add!(seq_filter, observable_measurement(Pup) * state_preparation(ρ_up), 2)
add!(seq_filter, TraceOut(), pt.nsteps)

prob_filter = evaluate_process(pt, seq_filter)

println("Probability of spin-up and reprepare at t=2:")
println(prob_filter)

@assert 0 <= real(prob_filter) <= 1 + 1e-10

# !!! note "ProductInstrument rules"
#     Multiply any two single-leg instruments with `*` when one targets the
#     output leg (`plev=0`) and one the input leg (`plev=1`). Indices can be
#     bound lazily at `add!` or supplied explicitly on each factor. Prepare the
#     initial state at time zero separately with `StatePreparation`, as above.

# ### [Two-time correlation preview](@id two-time-correlation-preview)
#
# Two-time correlations are built from instrument schedules too.
# The function `two_time_correlation_seq` constructs the instrument schedule for
#
# ```math
# \langle A(t_A)B(t_B)\rangle.
# ```
#
# Here we only show the usage pattern. A full discussion of multi-time
# correlations belongs in the examples section.

seq_corr = two_time_correlation_seq(
    pt,
    (Sz, 1),
    (Sz, 3);
    rho0=ρS0,
)

corr = evaluate_process(pt, seq_corr)

println("Two-time correlation preview:")
println(corr)

@assert corr isa ComplexF64
@assert isfinite(real(corr))

# ### Summary
#
# In this tutorial, we learned that:
#
# - a process tensor stores a multi-time open-system process,
# - `build_process_tensor` needs a system, an optional environment, `dt`, and
#   `nsteps`,
# - a bath mode stores its initial state, Hamiltonian, Liouville sites, and
#   system-mode coupling,
# - input/output time legs are exposed by `input_sites` and `output_sites`,
# - `evolve(pt, ρ0)` returns reduced density states over time,
# - `evaluate_process(pt, seq)` contracts a process tensor with an instrument
#   schedule,
# - fully closed schedules return scalars,
# - open-output schedules return state-like Liouville objects,
# - observables, collapse maps, and two-time correlations are all expressed
#   as instrument schedules,
# - lazy instruments defer materialization until contraction; use
#   `instrument_itensor` and `CustomTwoLegInstrument` for dense control,
#
# The examples section takes these same concepts further:
#
# - time-dependent Hamiltonians,
# - two-time and multi-time correlations,
# - multimode process tensors,
# - custom interventions and causal breaks.
