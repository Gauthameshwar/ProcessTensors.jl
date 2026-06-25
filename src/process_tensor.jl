import ITensorMPS: MPO as CoreMPO, MPS as CoreMPS, apply, linkdims, maxlinkdim
import ITensors: scalar, terms
import ITensors.Ops: Exact, Trotter
import Base: getproperty, setproperty!, show

"""
    ProcessTensor

Liouville-space MPO wrapper for a single-coupling-site process tensor.

Stores the environment influence and, by default, one-step system Liouvillian
propagation as an `MPO{Liouville}` core over time-ordered input/output legs.
Fields `system`, `environment`, `dt`, `nsteps`, and `coupling_site` record the
physical model. High-level APIs such as [`evaluate_process`](@ref),
[`evolve`](@ref), and [`two_time_correlation_seq`](@ref) expect
`embed_system_propagation = true`.

Unknown property access delegates to `.core`, like [`MPO`](@ref).

# Examples
```julia
pt = build_process_tensor(system, coupling_site; dt=0.1, nsteps=8)
trajectory = evolve(pt, ρ0)
```
"""
struct ProcessTensor{S<:AbstractSystem,E} <: AbstractMPO{Liouville}
    core::CoreMPO
    system::S
    environment::E
    dt::Float64
    nsteps::Int
    coupling_site::Index
    embed_system_propagation::Bool

    function ProcessTensor(
        core::CoreMPO,
        system::S,
        environment::E,
        dt::Real,
        nsteps::Integer,
        coupling_site::Index,
        embed_system_propagation::Bool=true,
    ) where {S<:AbstractSystem,E<:Union{Nothing,AbstractBath}}
        nsteps_int = Int(nsteps)
        nsteps_int >= 1 || throw(ArgumentError("A process tensor requires at least one timestep; got $nsteps."))
        _validate_coupling_site(system, coupling_site)
        length(core) == nsteps_int || throw(
            ArgumentError("ProcessTensor core length must equal nsteps for single-site PT. Got length(core)=$(length(core)) and nsteps=$nsteps_int."),
        )
        return new{S,E}(core, system, environment, float(dt), nsteps_int, coupling_site, embed_system_propagation)
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
    embed_system_propagation::Bool=true,
)
    length(system.sites) == 1 || throw(
        ArgumentError("ProcessTensor(core, system, environment, dt, nsteps) is only allowed for single-site systems. Pass coupling_site::Index explicitly."),
    )
    return ProcessTensor(core, system, environment, dt, nsteps, only(system.sites), embed_system_propagation)
end

function Base.getproperty(pt::ProcessTensor, sym::Symbol)
    sym in fieldnames(typeof(pt)) && return getfield(pt, sym)
    return getproperty(getfield(pt, :core), sym)
end

function Base.setproperty!(pt::ProcessTensor, sym::Symbol, val)
    sym in fieldnames(typeof(pt)) && return setfield!(pt, sym, val)
    return setproperty!(getfield(pt, :core), sym, val)
end


function Base.show(io::IO, pt::ProcessTensor)
    S, E = typeof(pt).parameters
    t_final = pt.dt * pt.nsteps
    χ = maxlinkdim(pt.core)
    print(io, pt.nsteps, "-step ProcessTensor{", nameof(S), ", ", nameof(E), "}")
    print(io, " | dt=", pt.dt, " | t_final=", round(t_final, digits=10), " | maxlinkdim=", χ)
    println(io)
    println(io)
    nsites = length(pt.system.sites)
    println(io, "  system:      ", nameof(S), "(nsites=", nsites, ", dissipative=", !isempty(pt.system.jump_ops), ")")
    print(io, "  environment: ")
    if pt.environment === nothing
        println(io, "none")
    elseif pt.environment isa AbstractBath
        env = pt.environment
        nmodes = length(env.modes)
        d_bath = isempty(env.sites) ? 1 : prod(dim.(env.sites))
        has_coupling = !isempty(terms(env.coupling)) ||
            any(!isempty(terms(m.coupling)) for m in env.modes)
        println(io, nameof(typeof(env)), "(nmodes=", nmodes, ", D_bath=", d_bath, ", coupling=", has_coupling, ")")
    else
        println(io, typeof(pt.environment))
    end
    ldims = Int[d for d in linkdims(pt.core) if d !== nothing]
    print(io, "  core:        MPO{Liouville}(length=", length(pt.core), ", linkdims=")
    if isempty(ldims)
        print(io, "none")
    else
        print(io, "[")
        if length(ldims) <= 10
            print(io, join(ldims, ", "))
        else
            print(io, join(ldims[1:3], ", "), ", …, ", ldims[end])
        end
        print(io, "]")
    end
    println(io, ")")
end

Base.show(io::IO, ::MIME"text/plain", pt::ProcessTensor) = show(io, pt)

function _tagset_with_tstep(s::Index, k::Int)
    tokens = filter(token -> token != "Site" && !startswith(token, "tstep="), tag_tokens(s))
    return ITensors.TagSet(join(vcat(tokens, ["tstep=$k"]), ","))
end

"""
    generate_pt_legs(site::Index, k::Int)

Construct the input/output Liouville legs for one process-tensor timestep.

The returned tuple is `(input_site, output_site)`. The input leg is the primed
version (`plev = 1`) of the output leg (`plev = 0`), and both carry the same
physical Liouville metadata as `site` plus a `tstep=k` tag.
"""
function generate_pt_legs(site::Index, k::Int)
    output_site = Index(dim(site); tags=_tagset_with_tstep(site, k))
    return prime(output_site), output_site
end

# Internal prime level for the system propagator fused into a core.
const _INTERNAL_PLEV = 2

# One-step embedded system Liouville map on PT legs `(in_k, out_k)`.
function _system_propagation_pt_core(
    system::AbstractSystem,
    in_k::Index,
    out_k::Index,
    dt::Real;
    alg=Trotter{2}(),
)
    U = liouvillian_propagator_itensor(
        system.H,
        system.sites,
        dt;
        alg=alg,
        jump_ops=system.jump_ops,
    )
    gate_site = only(system.sites)
    return replaceind(replaceind(U, prime(gate_site), in_k), gate_site, out_k)
end

function _require_embedded_propagation!(pt::ProcessTensor, caller::AbstractString)
    pt.embed_system_propagation && return
    @warn "$caller requires embed_system_propagation=true; lazy instrument evaluation is disabled for this ProcessTensor."
    throw(ArgumentError(
        "$caller: this ProcessTensor was built with embed_system_propagation=false. " *
        "Use manual instrument_itensor construction and ITensor contraction instead.",
    ))
end

# Build Markovian PT cores with identity or embedded system propagation on each `(in_k, out_k)` pair.
function _build_trivial_pt_cores(
    system::AbstractSystem,
    coupling_site::Index,
    dt::Real,
    nsteps::Int;
    embed_system_propagation::Bool=true,
    alg=Trotter{2}(),
)
    cores = ITensor[]
    inputs = Index[]
    outputs = Index[]
    for k in 0:(nsteps - 1)
        in_k, out_k = generate_pt_legs(coupling_site, k)
        push!(inputs, in_k)
        push!(outputs, out_k)
        if embed_system_propagation
            push!(cores, _system_propagation_pt_core(system, in_k, out_k, dt; alg=alg))
        else
            d = dim(in_k)
            identity_mat = Matrix{Float64}(I, d, d)
            push!(cores, ITensor(identity_mat, in_k, out_k))
        end
    end
    return cores
end

joint_liouville_dim(bath::AbstractBath, coupling_site::Index) =
    prod(dim.(collect(Index[vcat([only(m.sites) for m in bath.modes], [coupling_site])...])))

# Build one PT core per timestep by embedding a joint bath-system propagator and retaining one bath memory link.
function _build_bathmode_pt_cores(
    system::AbstractSystem,
    coupling_site::Index,
    bathmode::AbstractBathMode,
    dt::Real,
    nsteps::Int;
    bath_coupling::OpSum=OpSum(),
    alg=Exact(),
    embed_system_propagation::Bool=true,
    sys_alg=Trotter{2}(),
    kwargs...
)
    length(bathmode.sites) == 1 || throw(
        ArgumentError("build_process_tensor: AbstractBathMode must have exactly one site index. Got $(length(bathmode.sites)).")
    )
    env_liouv = only(bathmode.sites)
    d_env = dim(env_liouv)
    d_sys = dim(coupling_site)
    d_joint = d_env * d_sys
    _validate_dense_liouville_budget(d_joint; context="_build_bathmode_pt_cores")

    coupling_term = bathmode.coupling == OpSum() ? bath_coupling : bathmode.coupling
    # Joint physical Hamiltonian on [bath, system]; mode coupling uses sites 1=bath, 2=system.
    joint_ops = bathmode.H + coupling_term
    sites_vec = Index[env_liouv, coupling_site]
    U_ref = liouvillian_propagator_itensor(joint_ops, sites_vec, dt; alg=alg)

    # Bath virtual memory legs: nsteps cores use nsteps+1 links.
    bath_links = [Index(d_env; tags="PT,Link,tstep=$k") for k in 0:nsteps]

    cores = ITensor[]
    inputs = Index[]
    outputs = Index[]
    for k in 0:(nsteps - 1)
        in_k, out_k = generate_pt_legs(coupling_site, k)
        push!(inputs, in_k)
        push!(outputs, out_k)
        left = bath_links[k + 1]
        right = bath_links[k + 2]

        core_k = replaceind(U_ref, prime(env_liouv), right)
        core_k = replaceind(core_k, env_liouv, left)
        core_k = replaceind(core_k, prime(coupling_site), in_k)
        core_k = replaceind(core_k, coupling_site, out_k)
        push!(cores, core_k)
    end
    if embed_system_propagation
        for k in 0:(nsteps - 1)
            in_k = inputs[k + 1]
            out_k = outputs[k + 1]
            internal_out = prime(out_k, _INTERNAL_PLEV)
            core_k = replaceind(cores[k + 1], out_k, internal_out)
            sys_prop = replaceind(
                _system_propagation_pt_core(system, in_k, out_k, dt; alg=sys_alg),
                in_k,
                internal_out,
            )
            cores[k + 1] = core_k * sys_prop
        end
    end
    # Contract the first and last bath links with the initial bath state and the trace out
    initial_bath_state = instrument_itensor(StatePreparation(bathmode.rho0), [bath_links[1]'], 0)
    noprime!(initial_bath_state)
    trace_out = instrument_itensor(TraceOut(), [bath_links[end]], nsteps)

    cores[1] *= initial_bath_state
    cores[end] *= trace_out

    return cores
end

# Build PT cores for multiple modes using a fused bath memory link between timesteps.
function _build_multimode_pt_cores(
    system::AbstractSystem,
    coupling_site::Index,
    environment::AbstractBath,
    dt::Real,
    nsteps::Int;
    alg=Exact(),
    embed_system_propagation::Bool=true,
    sys_alg=Trotter{2}(),
    kwargs...
)
    isempty(environment.modes) && throw(ArgumentError("_build_multimode_pt_cores: environment must contain at least one mode."))

    modes = environment.modes
    nmodes = length(modes)
    sys_site = nmodes + 1
    sites_vec = Index[vcat([only(m.sites) for m in modes], [coupling_site])...]
    d_joint = prod(dim.(collect(sites_vec)))
    _validate_dense_liouville_budget(d_joint; context="_build_multimode_pt_cores")

    joint_ops = OpSum()
    for (i, mode) in enumerate(modes)
        for term in ITensors.terms(mode.H)
            c = ITensors.coefficient(term)
            args = Any[]
            for op_t in collect(last(term.args))
                push!(args, ITensors.name(op_t))
                for s in ITensors.sites(op_t)
                    src = Int(s)
                    src == 1 || throw(ArgumentError("_build_multimode_pt_cores(mode.H): expected local site 1, got $src."))
                    push!(args, i)
                end
            end
            joint_ops += (c, args...)
        end
        for term in ITensors.terms(mode.coupling)
            c = ITensors.coefficient(term)
            args = Any[]
            for op_t in collect(last(term.args))
                push!(args, ITensors.name(op_t))
                for s in ITensors.sites(op_t)
                    src = Int(s)
                    dst = if src == 1
                        i
                    elseif src == 2
                        sys_site
                    else
                        throw(ArgumentError("_build_multimode_pt_cores(mode.coupling): expected local sites 1 or 2, got $src."))
                    end
                    push!(args, dst)
                end
            end
            joint_ops += (c, args...)
        end
    end
    joint_ops += environment.coupling

    U_ref = liouvillian_propagator_itensor(joint_ops, sites_vec, dt; alg=alg)

    bath_sites = collect(sites_vec[1:(end - 1)])
    bath_sites_prime = prime.(bath_sites)
    comb_unprimed = combiner(bath_sites...; tags="PT,Link,FusedBath")
    comb_primed = combiner(bath_sites_prime...; tags="PT,Link,FusedBath,Prime")
    U_ref = U_ref * comb_unprimed * comb_primed

    fused_left = combinedind(comb_unprimed)
    fused_right = combinedind(comb_primed)
    d_bath = prod(dim.(bath_sites))
    bath_links = [Index(d_bath; tags="PT,Link,tstep=$k") for k in 0:nsteps]

    cores = ITensor[]
    inputs = Index[]
    outputs = Index[]
    for k in 0:(nsteps - 1)
        in_k, out_k = generate_pt_legs(coupling_site, k)
        push!(inputs, in_k)
        push!(outputs, out_k)
        left = bath_links[k + 1]
        right = bath_links[k + 2]

        core_k = replaceind(U_ref, prime(coupling_site), in_k)
        core_k = replaceind(core_k, coupling_site, out_k)
        core_k = replaceind(core_k, fused_left, left)
        core_k = replaceind(core_k, fused_right, right)
        push!(cores, core_k)
    end
    if embed_system_propagation
        for k in 0:(nsteps - 1)
            in_k = inputs[k + 1]
            out_k = outputs[k + 1]
            internal_out = prime(out_k, _INTERNAL_PLEV)
            core_k = replaceind(cores[k + 1], out_k, internal_out)
            sys_prop = replaceind(
                _system_propagation_pt_core(system, in_k, out_k, dt; alg=sys_alg),
                in_k,
                internal_out,
            )
            cores[k + 1] = core_k * sys_prop
        end
    end

    bath_state = ITensor(1.0)
    for mode in modes
        site = only(mode.sites)
        prep = instrument_itensor(StatePreparation(mode.rho0), Index[prime(site)], 0)
        noprime!(prep)
        hasind(prep, site) || throw(ArgumentError("_build_multimode_pt_cores: prepared mode state is missing mode site index."))
        bath_state *= prep
    end
    initial_bath_state = replaceind(bath_state * comb_unprimed, combinedind(comb_unprimed), bath_links[1])

    bath_trace = ITensor(1.0)
    for site in bath_sites_prime
        bath_trace *= instrument_itensor(TraceOut(; leg_plev=plev(site)), Index[site], nsteps)
    end
    trace_out = replaceind(bath_trace * comb_primed, combinedind(comb_primed), bath_links[end])

    cores[1] *= initial_bath_state
    cores[end] *= trace_out

    return cores
end

"""
    build_process_tensor(system, coupling_site; environment=nothing, dt, nsteps,
                         alg=Exact(), sys_alg=Trotter{2}(), embed_system_propagation=true)

Build a single-coupling-site process tensor.

`coupling_site` is the Liouville-space system leg kept as the process-tensor
input/output channel. Reuse the same Liouville index objects across the system,
bath, and instruments so later contractions match by exact index identity.

# Examples
```julia
pt = build_process_tensor(system, coupling_site; dt=0.1, nsteps=8)
```
"""
function build_process_tensor(
    system::AbstractSystem,
    coupling_site::Index;
    environment::Union{Nothing,AbstractBath}=nothing,
    dt::Real,
    nsteps::Integer,
    alg=Exact(),
    sys_alg=Trotter{2}(),
    embed_system_propagation::Bool=true,
)
    nsteps >= 1 || throw(ArgumentError("A process tensor requires at least one timestep; got $nsteps."))

    embed_in_cores = embed_system_propagation && length(system.sites) == 1

    cores = if environment === nothing
        _build_trivial_pt_cores(
            system,
            coupling_site,
            dt,
            nsteps;
            embed_system_propagation=embed_in_cores,
            alg=sys_alg,
        )
    else
        if joint_liouville_dim(environment, coupling_site) > MAX_DENSE_LIOUVILLE_DIM
            _validate_dense_liouville_budget(
                joint_liouville_dim(environment, coupling_site);
                context="build_process_tensor",
            )
        end
        if length(environment.modes) == 1
            _build_bathmode_pt_cores(
                system,
                coupling_site,
                environment.modes[1],
                dt,
                nsteps;
                bath_coupling=environment.coupling,
                alg=alg,
                embed_system_propagation=embed_in_cores,
                sys_alg=sys_alg,
            )
        else
            _build_multimode_pt_cores(
                system,
                coupling_site,
                environment,
                dt,
                nsteps;
                alg=alg,
                embed_system_propagation=embed_in_cores,
                sys_alg=sys_alg,
            )
        end
    end

    return ProcessTensor(CoreMPO(cores), system, environment, dt, nsteps, coupling_site, embed_in_cores)
end

"""
    build_process_tensor(system; environment=nothing, dt, nsteps,
                         alg=Exact(), sys_alg=Trotter{2}(), embed_system_propagation=true)

Build a process tensor for a single-site system by using its only Liouville
site as the coupling site.
"""
function build_process_tensor(
    system::AbstractSystem;
    environment::Union{Nothing,AbstractBath}=nothing,
    dt::Real,
    nsteps::Integer,
    alg=Exact(),
    sys_alg=Trotter{2}(),
    embed_system_propagation::Bool=true,
)
    length(system.sites) == 1 || throw(
        ArgumentError(
            "build_process_tensor(system; ...) is only allowed for single-site systems. " *
            "Pass `coupling_site::Index` explicitly for multi-site systems.",
        ),
    )
    return build_process_tensor(
        system, only(system.sites);
        environment=environment,
        dt=dt,
        nsteps=nsteps,
        alg=alg,
        sys_alg=sys_alg,
        embed_system_propagation=embed_system_propagation,
    )
end

_schedule_default_instr(::ProcessTensor) = IdentityOperation()

# Shared schedule validation for the lazy evaluation pipeline (create_instruments / evaluate_process / evolve):
# requires embedded propagation, optionally the schedule default, complete PT leg maps, and a single-leg tstep=0.
function _validate_instrument_schedule!(
    pt::ProcessTensor,
    seq::InstrumentSeq,
    default_instr::AbstractInstrument,
    caller::AbstractString;
    require_embedded::Bool=true,
    check_schedule_default::Bool=false,
)
    require_embedded && _require_embedded_propagation!(pt, caller)
    if check_schedule_default && pt.embed_system_propagation && default_instr isa SystemPropagation
        throw(
            ArgumentError(
                "$caller: system propagation is already embedded in this ProcessTensor; " *
                "use IdentityOperation() as the schedule default.",
            ),
        )
    end

    _, _, missing_in, missing_out = instrument_leg_maps(seq, pt.nsteps)
    isempty(missing_in) || throw(
        ArgumentError("$caller: missing input legs for tsteps $(missing_in)."),
    )
    isempty(missing_out) || throw(
        ArgumentError("$caller: missing output legs for tsteps $(missing_out)."),
    )
    resolve_instrument(seq, 0) isa SingleLegInstrument || throw(
        ArgumentError("$caller: tstep=0 must be a single-leg initial preparation."),
    )
    return nothing
end

# Tier B: default schedule with `IdentityOperation()` when propagation is embedded in `pt`.
function default_schedule(pt::ProcessTensor)
    _require_embedded_propagation!(pt, "default_schedule")
    return InstrumentSeq(default=_schedule_default_instr(pt), nsteps=pt.nsteps)
end

# Tier B: Liouville output leg (`plev=0`) at process-tensor time label `k`.
function output_sites(pt::ProcessTensor, k::Int)
    0 <= k < pt.nsteps || throw(BoundsError(0:(pt.nsteps - 1), k))
    core_k = pt.core[k + 1]
    sys_legs = Index[idx for idx in inds(core_k) if !has_tag_token(idx, "Link")]
    length(sys_legs) == 2 || throw(
        ArgumentError(
            "output_sites: core k=$k expected exactly 2 system legs, found $(length(sys_legs)).",
        ),
    )
    candidates = filter(
        idx -> plev(idx) == 0 &&
               tag_value(idx, "tstep=") == string(k) &&
               !has_tag_token(idx, "Link"),
        sys_legs,
    )
    length(candidates) == 1 || throw(
        ArgumentError("output_sites: expected one output leg at tstep=$k, found $(length(candidates))."),
    )
    out = only(candidates)
    plev(out) == 0 || throw(ArgumentError("output_sites: expected plev=0 output leg, got plev=$(plev(out))."))
    return Index[out]
end

# Tier B: Liouville input leg (`plev=1`) for slab with time label `tstep=k`.
function input_sites(pt::ProcessTensor, k::Int)
    0 <= k < pt.nsteps || throw(BoundsError(0:(pt.nsteps - 1), k))
    out = only(output_sites(pt, k))
    inn = prime(out)
    plev(inn) == 1 || throw(ArgumentError("input_sites: expected plev=1 input leg, got plev=$(plev(inn))."))
    inn in inds(pt.core[k + 1]) || throw(
        ArgumentError("input_sites: primed input leg at tstep=$k is not present on core k=$k."),
    )
    return Index[inn]
end

# Tier B: for evolve slot `step`, return `(out_prev, in_curr)` PT leg pair.
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

# Tier B (legacy): `(in_curr, out_prev)` — swapped `coupling_times` tuple order.
function coupling_sites(pt::ProcessTensor, step::Int)
    out_prev, in_curr = coupling_times(pt, step)
    return (in_curr, out_prev)
end

# Traces all indices except `keep`; for multimode PT this includes fused bath-memory links.
function _trace_out_except(t::ITensor, keep::AbstractVector{<:Index}; k::Int=0, environment=nothing)
    function _is_fused_bath_link(idx::Index, environment)
        environment isa AbstractBath || return false
        length(environment.modes) > 1 || return false
        "Link" in tag_tokens(idx) || return false
        return dim(idx) == prod(dim(only(mode.sites)) for mode in environment.modes)
    end
    
    function _fused_bath_trace_itensor(environment::AbstractBath, fused_link::Index, k::Int)
        bath_sites_prime = prime.([only(mode.sites) for mode in environment.modes])
        comb_primed = combiner(bath_sites_prime...; tags="PT,Link,FusedBath,Prime")
        bath_trace = ITensor(1.0)
        for site in bath_sites_prime
            bath_trace *= instrument_itensor(TraceOut(; leg_plev=plev(site)), Index[site], k)
        end
        return replaceind(bath_trace * comb_primed, combinedind(comb_primed), fused_link)
    end
    
    keep_vec = Index[keep...]
    out = t
    for idx in inds(out)
        idx in keep_vec && continue
        tstep_tag = tag_value(idx, "tstep=")
        idx_k = isnothing(tstep_tag) ? k : parse(Int, tstep_tag)
        trace_tensor = if _is_fused_bath_link(idx, environment)
            _fused_bath_trace_itensor(environment, idx, idx_k)
        else
            instrument_itensor(TraceOut(; leg_plev=plev(idx)), Index[idx], idx_k)
        end
        out *= trace_tensor
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

function _open_output_steps(seq::InstrumentSeq, nsteps::Int, default::AbstractInstrument)
    cuts = Int[]
    for step in 1:(nsteps - 1)
        resolve_instrument(seq, step, default) isa OpenOutput && push!(cuts, step)
    end
    return cuts
end

"""
    create_instruments(pt, seq; default, alg)

Materialize an `InstrumentSeq` as one `ITensor` per process-tensor timestep.

Most users call `evaluate_process` or `evolve` instead of building the
instrument tensors manually.
"""
function create_instruments(
    pt::ProcessTensor,
    seq::InstrumentSeq;
    default::AbstractInstrument=_schedule_default_instr(pt),
    alg=Trotter{2}(),
)
    _validate_instrument_schedule!(pt, seq, default, "create_instruments")

    prep = resolve_instrument(seq, 0)
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
                alg=alg,
            )
        elseif instr isa SingleLegInstrument
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

# Tier B: `true` when the schedule closes every PT leg (scalar `evaluate_process`).
"""
    all_pt_legs_contracted(pt, seq) -> Bool

Return `true` when the schedule closes every process-tensor leg, so
[`evaluate_process`](@ref) returns a `ComplexF64` scalar.
"""
function all_pt_legs_contracted(pt::ProcessTensor, seq::InstrumentSeq)
    !isempty(_open_output_steps(seq, pt.nsteps, seq.default)) && return false
    _, _, missing_in, missing_out = instrument_leg_maps(seq, pt.nsteps)
    isempty(missing_in) || return false
    isempty(missing_out) || return false
    final_instr = resolve_instrument(seq, pt.nsteps, seq.default)
    return final_instr isa TraceOut || final_instr isa SingleLegInstrument
end

"""
    evaluate_process(pt, seq; kwargs...) -> Union{ComplexF64, MPO{Liouville}}

Contract a process tensor with an instrument schedule.

Return a scalar when every process-tensor leg is closed, or an
`MPO{Liouville}` when one system output leg is left open.

# Examples
```julia
seq = InstrumentSeq(default=IdentityOperation(), nsteps=pt.nsteps)
add!(seq, StatePreparation(ρ0), 0)
add!(seq, TraceOut(), pt.nsteps)
result = evaluate_process(pt, seq)
```
"""
function evaluate_process(
    pt::ProcessTensor,
    seq::InstrumentSeq;
    default_instr::AbstractInstrument=_schedule_default_instr(pt),
    alg=Trotter{2}(),
    all_legs_contracted::Union{Nothing,Bool}=nothing,
)
    _validate_instrument_schedule!(pt, seq, default_instr, "evaluate_process"; check_schedule_default=true)

    open_cuts = _open_output_steps(seq, pt.nsteps, default_instr)
    open_keep_k = if isempty(open_cuts)
        nothing
    else
        length(open_cuts) == 1 || throw(
            ArgumentError(
                "evaluate_process: expected at most one OpenOutput in the schedule; found at steps $(open_cuts).",
            ),
        )
        open_cuts[1] - 1
    end
    instruments = create_instruments(pt, seq; default=default_instr, alg=alg)
    legs_closed = something(all_legs_contracted, all_pt_legs_contracted(pt, seq))

    result = pt.core[1] * instruments[1]
    for step in 1:(pt.nsteps - 1)
        instr = resolve_instrument(seq, step, default_instr)
        out_prev, in_curr = coupling_times(pt, step)
        if instr isa OpenOutput
            tmp = copy(pt.core[step + 1])
            tmp *= instrument_itensor(instr, in_curr, out_prev, step; dt=pt.dt, alg=alg)
            result *= tmp
        else
            result *= instruments[step + 1]
            result *= pt.core[step + 1]
        end
    end

    final_instr = resolve_instrument(seq, pt.nsteps, seq.default)
    if final_instr isa TraceOut || final_instr isa SingleLegInstrument
        out_prev, _ = coupling_times(pt, pt.nsteps)
        result *= instrument_itensor(final_instr, out_prev, pt.nsteps - 1)
    end

    if legs_closed
        n_open = length(inds(result))
        n_open == 0 || throw(
            ArgumentError(
                "evaluate_process: expected 0 uncontracted indices " *
                "(all_pt_legs_contracted=true) but found $n_open.",
            ),
        )
        return ComplexF64(scalar(result))
    end

    keep_k = something(open_keep_k, pt.nsteps - 1)
    keep = output_sites(pt, keep_k)
    reduced = if open_keep_k === nothing && length(inds(result)) == 1
        result
    else
        _trace_out_except(result, keep; k=keep_k, environment=pt.environment)
    end
    n_open = length(inds(reduced))
    n_open == 1 || throw(
        ArgumentError(
            "evaluate_process: expected exactly one open system leg at k=$keep_k, found $n_open.",
        ),
    )
    rho_liouv = _liouville_mps_from_itensor(reduced, keep)
    return MPO{Liouville}(CoreMPO(collect(rho_liouv.core)), rho_liouv.combiners)
end

"""
    evaluate_process(pt, seqs::AbstractVector{<:InstrumentSeq}; kwargs...) -> Vector{ComplexF64}

Evaluate a batch of fully contracted instrument schedules and return one scalar
per schedule.
"""
function evaluate_process(
    pt::ProcessTensor,
    seqs::AbstractVector{<:InstrumentSeq};
    kwargs...
)
    results = Vector{ComplexF64}(undef, length(seqs))
    for i in eachindex(seqs)
        val = evaluate_process(pt, seqs[i]; kwargs...)
        val isa ComplexF64 || throw(
            ArgumentError(
                "evaluate_process(batch): schedule at index $i is not fully contracted; " *
                "batch overload requires scalar schedules (all_pt_legs_contracted=true).",
            ),
        )
        results[i] = val
    end
    return results
end

"""
    evaluate_process(pt, rho0, seq; kwargs...)

Insert `StatePreparation(rho0)` at `tstep = 0` and contract the resulting
schedule with `pt`.
"""
function evaluate_process(
    pt::ProcessTensor,
    rho0,
    seq::InstrumentSeq;
    default_instr::AbstractInstrument=_schedule_default_instr(pt),
    kwargs...
)
    seq_full = InstrumentSeq(seq.default, seq.nsteps; entries=Dict{Int,AbstractInstrument}(pairs(seq.entries)))
    add!(seq_full, StatePreparation(rho0), 0)
    return evaluate_process(pt, seq_full; default_instr=default_instr, kwargs...)
end

"""
    evaluate_process(pt, rho0; kwargs...)

Evaluate a process tensor from an initial state using the default instrument
schedule.
"""
function evaluate_process(
    pt::ProcessTensor,
    rho0;
    default_instr::AbstractInstrument=_schedule_default_instr(pt),
    kwargs...
)
    seq = InstrumentSeq(default=default_instr, nsteps=pt.nsteps)
    add!(seq, StatePreparation(rho0), 0)
    return evaluate_process(pt, seq; default_instr=default_instr, kwargs...)
end

"""
    evolve(pt, seq; default_instr=_schedule_default_instr(pt), alg=Trotter{2}())

Return reduced system snapshots generated by contracting a process tensor with
an instrument schedule.

The result is a named tuple containing the sample times, Liouville-space
states, and reconstructed Hilbert-space density MPOs.

# Examples
```julia
trajectory = evolve(pt, ρ0)
ρ_t = trajectory.states_hilbert[3]
```
"""
function evolve(
    pt::ProcessTensor,
    seq::InstrumentSeq;
    default_instr::AbstractInstrument=_schedule_default_instr(pt),
    alg=Trotter{2}(),
    kwargs...
)
    _validate_instrument_schedule!(pt, seq, default_instr, "evolve"; check_schedule_default=true)

    instruments = create_instruments(pt, seq; default=default_instr, alg=alg)

    # Exactly `nsteps` reduced states: index `j` is the snapshot after PT slab `j-1`
    # (times `t = 0, dt, …, (nsteps-1)*dt`).
    states_liouville = Vector{MPS{Liouville}}(undef, pt.nsteps)
    states_hilbert = Vector{MPO{Hilbert}}(undef, pt.nsteps)
    times = [pt.dt * k for k in 0:(pt.nsteps - 1)]

    prev_pt_core = pt.core[1] * instruments[1]

    for k in 0:(pt.nsteps - 1)
        if k > 0
            step = k
            prev_pt_core *= instruments[step + 1]
            prev_pt_core *= pt.core[step + 1]
        end

        out_sites = output_sites(pt, k)
        reduced = _trace_out_except(prev_pt_core, out_sites; k=k, environment=pt.environment)
        rho_liouv = _liouville_mps_from_itensor(reduced, out_sites)
        states_liouville[k + 1] = rho_liouv
        states_hilbert[k + 1] = to_hilbert(rho_liouv)
    end

    return (times=times, states_liouville=states_liouville, states_hilbert=states_hilbert)
end

"""
    evolve(pt, rho0, seq; default_instr=_schedule_default_instr(pt))

Insert `StatePreparation(rho0)` at `tstep = 0` and return reduced system
snapshots for the resulting schedule.
"""
function evolve(
    pt::ProcessTensor,
    rho0,
    seq::InstrumentSeq;
    default_instr::AbstractInstrument=_schedule_default_instr(pt),
    kwargs...
)
    seq_full = InstrumentSeq(seq.default, seq.nsteps; entries=Dict{Int,AbstractInstrument}(pairs(seq.entries)))
    add!(seq_full, StatePreparation(rho0), 0)
    return evolve(pt, seq_full; default_instr=default_instr, kwargs...)
end

"""
    evolve(pt, rho0; default_instr=_schedule_default_instr(pt))

Return reduced system snapshots from an initial state using the default
instrument schedule.
"""
function evolve(
    pt::ProcessTensor,
    rho0;
    default_instr::AbstractInstrument=_schedule_default_instr(pt),
)
    seq = InstrumentSeq(default=default_instr, nsteps=pt.nsteps)
    add!(seq, StatePreparation(rho0), 0)
    return evolve(pt, seq; default_instr=default_instr)
end

"""
    two_time_correlation_seq(pt, (O_A, n_A), (O_B, n_B); rho0, default_instr)

Build an `InstrumentSeq` for the two-time correlator
``\\langle A(t_A) B(t_B)\\rangle``.

`rho0` is prepared at `tstep = 0`, and operator insertions are placed on the
process-tensor legs indexed by `n_A` and `n_B`.

# Examples
```julia
seq = two_time_correlation_seq(pt, (O_A, 1), (O_B, 3); rho0=ρ0)
result = evaluate_process(pt, seq)
```
"""
function two_time_correlation_seq(
    pt::ProcessTensor,
    op_a::Tuple{OpSum, Int},
    op_b::Tuple{OpSum, Int};
    rho0::Union{AbstractMPO{Hilbert}, AbstractMPS{Hilbert}},
    default_instr::AbstractInstrument=_schedule_default_instr(pt),
)
    _require_embedded_propagation!(pt, "two_time_correlation_seq")
    O_A, n_A = op_a
    O_B, n_B = op_b
    n_A >= 0 || throw(ArgumentError("two_time_correlation_seq: time index n_A must be ≥ 0; got $n_A."))
    n_B >= 0 || throw(ArgumentError("two_time_correlation_seq: time index n_B must be ≥ 0; got $n_B."))
    n_late = max(n_A, n_B)
    n_late + 1 <= pt.nsteps || throw(
        ArgumentError(
            "two_time_correlation_seq: max(n_A, n_B) + 1 = $(n_late + 1) exceeds pt.nsteps=$(pt.nsteps).",
        ),
    )

    phys_sites = _phys_sites_from_hilbert_state(rho0)
    n_early = min(n_A, n_B)
    slot_late = n_late + 1
    terminal_late = slot_late == pt.nsteps

    seq = InstrumentSeq(default=default_instr, nsteps=pt.nsteps)

    if n_A == n_B
        add!(seq, StatePreparation(rho0), 0)
        # Terminal single-leg and interior left_action compose observables in opposite order;
        # B*A factors → Tr(A B ρ) via left_action, A*B factors → Tr(A B ρ) on the terminal leg.
        same_time = terminal_late ?
                    ObservableMeasurement(O_A) * ObservableMeasurement(O_B) :
                    ObservableMeasurement(O_B) * ObservableMeasurement(O_A)
        if terminal_late
            add!(seq, same_time, pt.nsteps)
        else
            add!(seq, left_action(same_time, phys_sites), slot_late)
            for step in (slot_late + 1):(pt.nsteps - 1)
                add!(seq, IdentityOperation(), step)
            end
            add!(seq, TraceOut(), pt.nsteps)
        end
    else
        if n_A > n_B
            O_early, O_late, early_side = O_B, O_A, :left
        else
            O_early, O_late, early_side = O_A, O_B, :right
        end

        if n_early == 0
            prep = early_side === :left ?
                   ObservableMeasurement(O_early; leg_plev=1) * StatePreparation(rho0) :
                   StatePreparation(rho0) * ObservableMeasurement(O_early; leg_plev=1)
            add!(seq, prep, 0)
        else
            add!(seq, StatePreparation(rho0), 0)
            early_slot = n_early + 1
            early_lr = early_side === :left ? left_action(O_early, phys_sites) : right_action(O_early, phys_sites)
            add!(seq, early_lr, early_slot)
        end

        if terminal_late
            add!(seq, ObservableMeasurement(O_late), pt.nsteps)
        else
            add!(seq, left_action(O_late, phys_sites), slot_late)
            for step in (slot_late + 1):(pt.nsteps - 1)
                add!(seq, IdentityOperation(), step)
            end
            add!(seq, TraceOut(), pt.nsteps)
        end
    end

    return seq
end
