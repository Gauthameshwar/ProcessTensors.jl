import ITensorMPS: MPO as CoreMPO, MPS as CoreMPS, apply
import Base: getproperty, setproperty!
using LinearAlgebra: exp, kron

struct ProcessTensor{S<:AbstractSystem,E} <: AbstractMPO{Liouville}
    core::CoreMPO
    system::S
    environment::E
    dt::Float64
    nsteps::Int
    coupling_site::Index

    function ProcessTensor(
        core::CoreMPO,
        system::S,
        environment::E,
        dt::Real,
        nsteps::Integer,
        coupling_site::Index,
    ) where {S<:AbstractSystem,E<:Union{Nothing,AbstractBath}}
        nsteps_int = Int(nsteps)
        nsteps_int >= 1 || throw(ArgumentError("A process tensor requires at least one timestep; got $nsteps."))
        _validate_coupling_site(system, coupling_site)
        length(core) == nsteps_int || throw(
            ArgumentError("ProcessTensor core length must equal nsteps for single-site PT. Got length(core)=$(length(core)) and nsteps=$nsteps_int."),
        )
        return new{S,E}(core, system, environment, float(dt), nsteps_int, coupling_site)
    end
end

function _validate_coupling_site(system::AbstractSystem, coupling_site::Index)
    isempty(system.sites) && throw(ArgumentError("System sites cannot be empty."))
    coupling_site in system.sites || throw(
        ArgumentError("coupling_site must be one of system.sites in Liouville space."),
    )
    return nothing
end

function ProcessTensor(
    core::CoreMPO,
    system::AbstractSystem,
    environment,
    dt::Real,
    nsteps::Integer,
)
    length(system.sites) == 1 || throw(
        ArgumentError("ProcessTensor(core, system, environment, dt, nsteps) is only allowed for single-site systems. Pass coupling_site::Index explicitly."),
    )
    return ProcessTensor(core, system, environment, dt, nsteps, only(system.sites))
end

function Base.getproperty(pt::ProcessTensor, sym::Symbol)
    sym in fieldnames(typeof(pt)) && return getfield(pt, sym)
    return getproperty(getfield(pt, :core), sym)
end

function Base.setproperty!(pt::ProcessTensor, sym::Symbol, val)
    sym in fieldnames(typeof(pt)) && return setfield!(pt, sym, val)
    return setproperty!(getfield(pt, :core), sym, val)
end

function _tagset_with_tstep(s::Index, k::Int)
    tokens = filter(token -> token != "Site" && !startswith(token, "tstep="), tag_tokens(s))
    return ITensors.TagSet(join(vcat(tokens, ["tstep=$k"]), ","))
end

function generate_pt_legs(site::Index, k::Int)
    output_site = Index(dim(site); tags=_tagset_with_tstep(site, k))
    return prime(output_site), output_site
end

"""Build a trivial process tensor for a single site system, i.e. a sequence of identity operators on the input and output legs.
This is useful for a Markovian system where the system is not coupled to any bath modes.
"""
function _build_trivial_pt_cores(coupling_site::Index, nsteps::Int)
    cores = ITensor[]
    for k in 0:(nsteps - 1)
        in_k, out_k = generate_pt_legs(coupling_site, k)
        push!(cores, delta(in_k, out_k))
    end
    return cores
end

"""Build one Liouville-space PT core per timestep for a single bath mode by embedding a one-step joint bath-system propagator onto `(in_k, out_k, link_k, link_{k+1})`."""
function _build_bathmode_pt_cores(coupling_site::Index, coupling_term::OpSum, bathmode::AbstractBathMode, spectral_density::AbstractSpectralDensity, dt::Real, nsteps::Int; kwargs...)
    nsteps >= 1 || throw(ArgumentError("_build_bathmode_pt_cores: nsteps must be at least 1; got $nsteps."))
    length(bathmode.sites) == 1 || throw(
        ArgumentError("build_process_tensor: AbstractBathMode must have exactly one site index. Got $(length(bathmode.sites)).")
    )
    env_liouv = only(bathmode.sites)
    d_env = dim(env_liouv)
    d_sys = dim(coupling_site)

    # Build joint Liouvillian MPO on the two-site ordering [bath, system].
    joint_ops = bathmode.H + coupling_term # should also contain the spectral density term, where the omega corresponds to the free energy-level spacing of the bath mode
    joint_L = MPO_Liouville(joint_ops, Index[env_liouv, coupling_site])

    # Materialize dense one-step propagator U_ref = exp(dt * L) once, outside the time loop.
    Lj = foldl(*, joint_L)

    L_site_order = [prime(env_liouv), prime(coupling_site), env_liouv, coupling_site]
    L_dense_tensor = Array(Lj, L_site_order...)
    d_joint = d_env * d_sys
    L_dense = reshape(ComplexF64.(L_dense_tensor), d_joint, d_joint)
    U_dense = exp(float(dt) * L_dense)
    U_ref = ITensor(
        reshape(U_dense, d_env, d_sys, d_env, d_sys),
        L_site_order...
    )

    # Bath virtual memory legs: nsteps cores use nsteps+1 links.
    bath_links = [Index(d_env; tags="PT,Link,tstep=$k") for k in 0:nsteps]

    cores = ITensor[]
    for k in 0:(nsteps - 1)
        in_k, out_k = generate_pt_legs(coupling_site, k)
        left = bath_links[k + 1]
        right = bath_links[k + 2]

        core_k = replaceind(U_ref, prime(coupling_site), in_k)
        core_k = replaceind(core_k, coupling_site, out_k)
        core_k = replaceind(core_k, env_liouv, left)
        core_k = replaceind(core_k, prime(env_liouv), right)
        push!(cores, core_k)
    end

    # Contract the first and last bath links with the initial bath state and the trace out
    initial_bath_state = instrument_itensor(StatePreparation(bathmode.rho0), [bath_links[1]'], 0)
    noprime!(initial_bath_state)
    trace_out = instrument_itensor(TraceOut(), [bath_links[end]], nsteps)

    cores[1] *= initial_bath_state
    cores[end] *= trace_out

    return cores
end

function build_process_tensor(
    system::AbstractSystem,
    coupling_site::Index;
    environment::Union{Nothing,AbstractBath}=nothing,
    dt::Real,
    nsteps::Integer,
)
    nsteps >= 1 || throw(ArgumentError("A process tensor requires at least one timestep; got $nsteps."))
    _validate_coupling_site(system, coupling_site)

    cores = if environment === nothing
        _build_trivial_pt_cores(coupling_site, nsteps)
    else
        _build_bathmode_pt_cores(coupling_site, environment.coupling, environment.modes[1], environment.spectral_density, dt, nsteps)
    end

    return ProcessTensor(CoreMPO(cores), system, environment, dt, nsteps, coupling_site)
end

# Single-site convenience: defaults `coupling_site` to the system's only site.
function build_process_tensor(
    system::AbstractSystem;
    environment::Union{Nothing,AbstractBath}=nothing,
    dt::Real,
    nsteps::Integer,
)
    length(system.sites) == 1 || throw(
        ArgumentError(
            "build_process_tensor(system; ...) is only allowed for single-site systems. " *
            "Pass `coupling_site::Index` explicitly for multi-site systems.",
        ),
    )
    return build_process_tensor(
        system, only(system.sites);
        environment=environment, dt=dt, nsteps=nsteps,
    )
end

# Default schedule uses InstrumentSeq with SystemPropagation fallback.
default_schedule(pt::ProcessTensor) = InstrumentSeq(default=SystemPropagation(pt.system), nsteps=pt.nsteps)

"""
    output_sites(pt, k) -> Vector{Index}

Liouville **output** leg (unprimed, `plev=0`) at process-tensor time label `k`.
Valid `k`: `0:(pt.nsteps - 1)`. After a full trajectory, the reduced state attaches to
`output_sites(pt, pt.nsteps - 1)` once bath / memory legs are contracted.
"""
function output_sites(pt::ProcessTensor, k::Int)
    0 <= k < pt.nsteps || throw(BoundsError(0:(pt.nsteps - 1), k))
    core_k = pt.core[k + 1]
    candidates = filter(
        idx -> plev(idx) == 0 &&
               tag_value(idx, "tstep=") == string(k) &&
               !has_tag_token(idx, "Link"),
        inds(core_k),
    )
    length(candidates) == 1 || throw(
        ArgumentError("output_sites: expected one output leg at tstep=$k, found $(length(candidates))."),
    )
    out = only(candidates)
    plev(out) == 0 || throw(ArgumentError("output_sites: expected plev=0 output leg, got plev=$(plev(out))."))
    return Index[out]
end

"""
    input_sites(pt, k) -> Vector{Index}

Liouville **input** leg (primed, `plev=1`) for the slab with time label **`tstep=k`**,
with **`k ∈ 0:(pt.nsteps - 1)`** (initialization attaches to **`k = 0`**, i.e. `tstep=0'`).
"""
function input_sites(pt::ProcessTensor, k::Int)
    0 <= k < pt.nsteps || throw(BoundsError(0:(pt.nsteps - 1), k))
    inn = prime(only(output_sites(pt, k)))
    plev(inn) == 1 || throw(ArgumentError("input_sites: expected plev=1 input leg, got plev=$(plev(inn))."))
    return Index[inn]
end

"""
    coupling_times(pt, step) -> (out_prev, in_curr)

For evolve slot **`step ∈ 1:pt.nsteps`**, return **`(out_prev, in_curr)`** for the two-leg
instrument that advances the system line **after** slab `step - 1` and **before** slab `step`
(in 1-based evolve counting):

  * `out_prev = output_sites(pt, step - 1)` — unprimed, `tstep = step-1`
  * `in_curr` — primed, `tstep = step` (from `generate_pt_legs(..., step)[1]`; for `step == pt.nsteps`
    this is the terminal primed leg with `tstep = pt.nsteps`, satisfying `tin == tout + 1` in
    `Instruments._validate_two_leg_map`).

When calling `instrument_itensor` for `SystemPropagation` / `IdentityOperation`, pass
`(input_pt_sites=in_curr, output_pt_sites=out_prev)`.
"""
function coupling_times(pt::ProcessTensor, step::Int)
    1 <= step <= pt.nsteps || throw(BoundsError(1:pt.nsteps, step))
    if step <= pt.nsteps - 1
        inn = only(input_sites(pt, step))
    else
        # Terminal synthetic input leg (used only for compatibility checks / boundary maps).
        inn = generate_pt_legs(pt.coupling_site, step)[1]
    end
    plev(inn) == 1 || throw(ArgumentError("coupling_times: expected plev=1 on in_curr, got plev=$(plev(inn))."))
    return (output_sites(pt, step - 1), Index[inn])
end

"""Legacy alias: `(in_curr, out_prev)` — same indices as `coupling_times(pt, step)`, swapped tuple order. This will be removed once we have a stable implementation of the build and evolve of process tensors without this function"""
function coupling_sites(pt::ProcessTensor, step::Int)
    out_prev, in_curr = coupling_times(pt, step)
    return (in_curr, out_prev)
end

# This function assumes the memory links have d^2 dim. This is only true for the single mode bath, and can fail otherwise. Change this in the future release of the package.
function _trace_out_except(t::ITensor, keep::AbstractVector{<:Index}; k::Int=0)
    keep_vec = Index[keep...]
    out = t
    for idx in inds(out)
        idx in keep_vec && continue
        tstep_tag = tag_value(idx, "tstep=")
        idx_k = isnothing(tstep_tag) ? k : parse(Int, tstep_tag)
        out *= instrument_itensor(TraceOut(; leg_plev=plev(idx)), Index[idx], idx_k)
    end
    return out
end

# Wrap a contracted ITensor core (reduced density matrix vector) into an MPS{Liouville} object
function _liouville_mps_from_itensor(t::ITensor, liouv_sites::AbstractVector{<:Index})
    length(liouv_sites) == 1 || throw(ArgumentError("_liouville_mps_from_itensor currently supports a single Liouville site."))
    liouv_site = only(liouv_sites)
    t_loc = t
    if !hasind(t_loc, liouv_site)
        hasind(t_loc, prime(liouv_site)) || throw(
            ArgumentError("_liouville_mps_from_itensor: reduced tensor is missing Liouville site $(liouv_site)."),
        )
        t_loc = replaceind(t_loc, prime(liouv_site), liouv_site)
    end
    phys_site = _phys_site_from_liouv(liouv_site)
    d = dim(phys_site)
    d2 = dim(liouv_site)
    d * d == d2 || throw(ArgumentError("_liouville_mps_from_itensor expects dim(liouv_site)=d^2."))

    # Convert PT's local Liouville basis ordering to the package's canonical ordering
    # used by to_liouville/to_hilbert by applying vec(ρ) -> vec(ρᵀ).
    perm = zeros(ComplexF64, d2, d2)
    for i in 1:d, j in 1:d
        old = (i - 1) * d + j
        new = (j - 1) * d + i
        perm[new, old] = 1.0
    end
    transpose_map = ITensor(perm, prime(liouv_site), liouv_site)
    t_loc = transpose_map * t_loc
    t_loc = replaceind(t_loc, prime(liouv_site), liouv_site)

    comb = combiner(phys_site, prime(phys_site); tags=tags(liouv_site))
    comb = replaceind(comb, combinedind(comb), liouv_site)
    return MPS{Liouville}(CoreMPS([t_loc]), ITensor[comb])
end

# User-facing function to create an instrument schedule from a process tensor and an instrument sequence
function create_instruments(
    pt::ProcessTensor,
    seq::InstrumentSeq;
    default::AbstractInstrument=SystemPropagation(pt.system),
    order::Int=2,
)
    in_map, out_map, missing_in, missing_out = instrument_leg_maps(seq, pt.nsteps)
    isempty(missing_in) || throw(
        ArgumentError("create_instruments: missing input legs for tsteps $(missing_in)."),
    )
    isempty(missing_out) || throw(
        ArgumentError("create_instruments: missing output legs for tsteps $(missing_out)."),
    )

    prep = resolve_instrument(seq, 0)
    prep isa StatePreparation || throw(
        ArgumentError("create_instruments: tstep=0 must be a StatePreparation."),
    )

    instruments = Vector{ITensor}(undef, pt.nsteps)
    prep_in = input_sites(pt, 0)
    instruments[1] = instrument_itensor(prep, prep_in, 0)

    for step in 1:(pt.nsteps - 1)
        instr = resolve_instrument(seq, step, default)
        out_prev, in_curr = coupling_times(pt, step)
        if instr isa TwoLegInstrument
            instruments[step + 1] = instrument_itensor(
                instr,
                in_curr,
                out_prev,
                step;
                dt=pt.dt,
                order=order,
            )
        elseif instr isa ObservableMeasurement || instr isa TraceOut
            if instr.leg_plev == 0
                instruments[step + 1] = instrument_itensor(instr, out_prev, step)
            else
                instruments[step + 1] = instrument_itensor(instr, in_curr, step)
            end
        else
            throw(ArgumentError("create_instruments: unsupported instrument $(typeof(instr)) at step=$step."))
        end
    end

    return instruments
end


# Evolve a process tensor 
function evolve(
    pt::ProcessTensor,
    seq::InstrumentSeq;
    default_instr::AbstractInstrument=SystemPropagation(pt.system),
    order::Int=2,
    kwargs...
)
    in_map, out_map, missing_in, missing_out = instrument_leg_maps(seq, pt.nsteps)
    isempty(missing_in) || throw(
        ArgumentError("evolve: missing input legs for tsteps $(missing_in)."),
    )
    isempty(missing_out) || throw(
        ArgumentError("evolve: missing output legs for tsteps $(missing_out)."),
    )

    instruments = create_instruments(pt, seq; default=default_instr, order=order)

    # Exactly `nsteps` reduced states: index `j` is the snapshot after PT slab `j-1`
    # (times `t = 0, dt, …, (nsteps-1)*dt`).
    states_liouville = Vector{MPS{Liouville}}(undef, pt.nsteps)
    states_hilbert = Vector{MPO{Hilbert}}(undef, pt.nsteps)
    times = [pt.dt * k for k in 0:(pt.nsteps - 1)]

    resolve_instrument(seq, 0) isa StatePreparation ||
        throw(ArgumentError("evolve: tstep=0 must be StatePreparation."))

    prev_pt_core = pt.core[1] * instruments[1]

    for k in 0:(pt.nsteps - 1)
        if k > 0
            step = k
            prev_pt_core *= instruments[step + 1]
            prev_pt_core *= pt.core[step + 1]
        end

        out_sites = output_sites(pt, k)
        reduced = _trace_out_except(prev_pt_core, out_sites; k=k)
        rho_liouv = _liouville_mps_from_itensor(reduced, out_sites)
        states_liouville[k + 1] = rho_liouv
        states_hilbert[k + 1] = to_hilbert(rho_liouv)
    end

    return (times=times, states_liouville=states_liouville, states_hilbert=states_hilbert)
end

# Evolve a process tensor pt with an initial state and a defined instrument sequence
function evolve(
    pt::ProcessTensor,
    rho0,
    seq::InstrumentSeq;
    default_instr::AbstractInstrument=SystemPropagation(pt.system),
    kwargs...
)
    seq_full = InstrumentSeq(seq.default, seq.nsteps; entries=Dict{Int,AbstractInstrument}(pairs(seq.entries)))
    add!(seq_full, StatePreparation(rho0), 0)
    return evolve(pt, seq_full; default_instr=default_instr, kwargs...)
end

# Evolve a process tensor pt with an initial state rho0 and the default instrument sequence
function evolve(
    pt::ProcessTensor,
    rho0;
    default_instr::AbstractInstrument=SystemPropagation(pt.system),
)
    seq = InstrumentSeq(default=default_instr, nsteps=pt.nsteps)
    add!(seq, StatePreparation(rho0), 0)
    return evolve(pt, seq; default_instr=default_instr)
end
