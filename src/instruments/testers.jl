# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: src/instruments/testers.jl
# Contributor: Gauthameshwar S.
#
# Defines memory-bearing tester ancillas, tester actions, and intervention
# schedules used during process-tensor contraction.

# Return the physical local dimension and SiteType encoded by a Hilbert or
# Liouville site. Actions are compared by physical meaning, not Index identity.
function _tester_site_signature(site::Index)
    family = _liouv_site_type(tag_tokens(site))
    family === nothing && throw(
        ArgumentError("Could not infer a physical SiteType from action site $(site)."),
    )
    if has_tag_token(site, "Liouv")
        d2 = dim(site)
        d = isqrt(d2)
        d * d == d2 || throw(
            ArgumentError("Liouville action site $(site) has dim=$d2, expected a perfect square."),
        )
        return d, family
    end
    return dim(site), family
end

function _collect_tester_sites(
    owner::AbstractString,
    sites::AbstractVector{<:Index};
    count::Int=1,
    hilbert_only::Bool=false,
)
    collected = Index[sites...]
    length(collected) == count || throw(
        ArgumentError("$owner: expected exactly $count site index$(count == 1 ? "" : "es"); got $(length(collected))."),
    )
    liouv_flags = map(site -> has_tag_token(site, "Liouv"), collected)
    any(liouv_flags) && !all(liouv_flags) && throw(
        ArgumentError("$owner: sites must be either all Hilbert or all Liouville indices."),
    )
    hilbert_only && any(liouv_flags) && throw(
        ArgumentError("$owner: unitary data requires Hilbert-space sites."),
    )
    return collected
end

function _validate_tester_initial_state(
    state::AbstractMPS{Liouville},
    memory_sites::AbstractVector{<:Index},
)
    state_sites = Index[siteinds(state)...]
    state_sites == memory_sites || throw(
        ArgumentError(
            "TesterMemory: Liouville rho0 must use the exact memory site indices. " *
            "Got $(state_sites) and $(memory_sites).",
        ),
    )
    return nothing
end

function _validate_tester_initial_state(
    state::Union{AbstractMPS{Hilbert},AbstractMPO{Hilbert}},
    memory_sites::AbstractVector{<:Index},
)
    state_sites = if state isa AbstractMPO{Hilbert}
        _phys_sites_from_hilbert_mpo(state)
    else
        Index[siteinds(state)...]
    end
    length(state_sites) == length(memory_sites) || throw(
        ArgumentError(
            "TesterMemory: rho0 has $(length(state_sites)) physical sites, expected $(length(memory_sites)).",
        ),
    )
    for (state_site, memory_site) in zip(state_sites, memory_sites)
        _tester_site_signature(state_site) == _tester_site_signature(memory_site) || throw(
            ArgumentError(
                "TesterMemory: rho0 site $(state_site) is incompatible with memory site $(memory_site).",
            ),
        )
    end
    return nothing
end

"""
    TesterMemory(sites, rho0)

Memory-bearing ancillary quantum object used by a process-tensor tester.

`sites` may contain one Hilbert- or Liouville-space site of any physical
SiteType; Hilbert input is normalized with [`liouv_sites`](@ref). `rho0` is the
initial memory state and must match the memory site's physical dimension and
SiteType. A `TesterMemory` stores no Hamiltonian or intervention schedule.

Use [`tester_memory`](@ref) for ordinary construction.
"""
struct TesterMemory{S<:Union{AbstractMPS,AbstractMPO{Hilbert}}} <: AbstractInstrument
    sites::Vector{Index}
    rho0::S

    function TesterMemory(
        sites::AbstractVector{<:Index},
        rho0::S,
    ) where {S<:Union{AbstractMPS,AbstractMPO{Hilbert}}}
        collected = _collect_tester_sites("TesterMemory", sites)
        # Store Liouville sites; Hilbert input is converted with `liouv_sites`.
        memory_sites = has_tag_token(only(collected), "Liouv") ? collected : liouv_sites(collected)
        _validate_tester_initial_state(rho0, memory_sites)
        return new{S}(memory_sites, rho0)
    end
end

"""
    tester_memory(sites, rho0)

Construct a single-site [`TesterMemory`](@ref) ancillary object.

# Examples
```julia
s = siteinds("Qubit", 1)
rho0 = to_dm(MPS(s, ["0"]))
memory = tester_memory(s, rho0)
```
"""
tester_memory(sites::AbstractVector{<:Index}, rho0) = TesterMemory(sites, rho0)

"""
    AbstractTesterAction

Abstract interface for tester actions on memory alone or jointly on system and memory.

Actions retain Hilbert-site metadata for later binding to runtime
process-tensor and tester-memory legs.
"""
abstract type AbstractTesterAction end

"""
    TesterIdentity

Identity action carrying tester memory forward without local evolution.
"""
struct TesterIdentity <: AbstractTesterAction end

"""
    TesterPropagation

Hamiltonian description for propagation of tester memory alone.
"""
struct TesterPropagation{H} <: AbstractTesterAction
    H::H
    sites::Vector{Index}

    function TesterPropagation(H::T, sites::AbstractVector{<:Index}) where {T}
        validated = _collect_tester_sites("TesterPropagation", sites)
        return new{T}(H, validated)
    end
end

"""
    TesterUnitary

Ready-made Hilbert-space unitary acting on the memory site.
"""
struct TesterUnitary <: AbstractTesterAction
    U::ITensor
    sites::Vector{Index}

    function TesterUnitary(U::ITensor, sites::AbstractVector{<:Index})
        validated = _collect_tester_sites("TesterUnitary", sites; hilbert_only=true)
        _validate_unitary_sites("TesterUnitary", U, validated)
        return new(U, validated)
    end
end

"""
    JointPropagation

Hamiltonian description for joint system-memory propagation.
"""
struct JointPropagation{H} <: AbstractTesterAction
    H::H
    system_sites::Vector{Index}
    tester_sites::Vector{Index}

    function JointPropagation(
        H::T,
        system_sites::AbstractVector{<:Index},
        tester_sites::AbstractVector{<:Index},
    ) where {T}
        system = _collect_tester_sites("JointPropagation system", system_sites)
        memory = _collect_tester_sites("JointPropagation memory", tester_sites)
        _validate_disjoint_action_sites("JointPropagation", system, memory)
        return new{T}(H, system, memory)
    end
end

"""
    JointUnitary

Ready-made Hilbert-space unitary acting jointly on system and memory sites.
"""
struct JointUnitary <: AbstractTesterAction
    U::ITensor
    system_sites::Vector{Index}
    tester_sites::Vector{Index}

    function JointUnitary(
        U::ITensor,
        system_sites::AbstractVector{<:Index},
        tester_sites::AbstractVector{<:Index},
    )
        system = _collect_tester_sites("JointUnitary system", system_sites; hilbert_only=true)
        memory = _collect_tester_sites("JointUnitary memory", tester_sites; hilbert_only=true)
        _validate_disjoint_action_sites("JointUnitary", system, memory)
        _validate_unitary_sites("JointUnitary", U, vcat(system, memory))
        return new(U, system, memory)
    end
end

function _validate_unitary_sites(
    owner::AbstractString,
    U::ITensor,
    sites::AbstractVector{<:Index},
)
    unitary_inds = inds(U)
    for site in sites
        site in unitary_inds || throw(
            ArgumentError("$owner: unitary is missing Hilbert input index $(site)."),
        )
        prime(site) in unitary_inds || throw(
            ArgumentError("$owner: unitary is missing Hilbert output index $(prime(site))."),
        )
    end
    length(unitary_inds) == 2 * length(sites) || throw(
        ArgumentError(
            "$owner: unitary must contain exactly one input/output pair per supplied site; " *
            "got $(length(unitary_inds)) indices for $(length(sites)) sites.",
        ),
    )
    return nothing
end

function _validate_disjoint_action_sites(
    owner::AbstractString,
    system_sites::AbstractVector{<:Index},
    tester_sites::AbstractVector{<:Index},
)
    any(system_site == tester_site for system_site in system_sites for tester_site in tester_sites) &&
        throw(ArgumentError("$owner: system and memory sites must be distinct indices."))
    return nothing
end

"""
    tester_identity()

Construct the default action that carries tester memory forward unchanged.

# Examples
```julia
seq = TesterSeq(default=tester_identity(), nsteps=4)
```
"""
tester_identity() = TesterIdentity()

"""
    tester_propagation(H::OpSum, sites)

Store a Hamiltonian for later propagation of the memory alone on `sites`.

The Hamiltonian is stored and materialized during
[`ProcessTensors.evaluate_process`](@ref).

# Examples
```julia
s = siteinds("Qubit", 1)
H = OpSum() + (0.2, "X", 1)
action = tester_propagation(H, s)
```
"""
tester_propagation(H::OpSum, sites::AbstractVector{<:Index}) =
    TesterPropagation(H, Index[sites...])

"""
    tester_unitary(U::ITensor, sites)

Store a ready-made Hilbert-space unitary acting on the memory alone.

`U` must contain each supplied site as an unprimed input index and the
corresponding primed output index.

# Examples
```julia
s = siteinds("Qubit", 1)
action = tester_unitary(op("X", only(s)), s)
```
"""
tester_unitary(U::ITensor, sites::AbstractVector{<:Index}) =
    TesterUnitary(U, Index[sites...])

"""
    joint_propagation(H::OpSum, system_sites, tester_sites)

Store a joint system-memory Hamiltonian for later propagation.

The `tester_sites` are the tester's ancillary memory sites. OpSum site numbers
refer to `vcat(system_sites, tester_sites)`. For the initial
one-system-site/one-memory-site implementation, site `1` is the system and site
`2` is the memory. The Hamiltonian is stored and materialized during
[`ProcessTensors.evaluate_process`](@ref).

# Examples
```julia
s_system = siteinds("Qubit", 1)
s_tester = siteinds("Qubit", 1)
H = OpSum() + (0.1, "Z", 1, "Z", 2)
action = joint_propagation(H, s_system, s_tester)
```
"""
joint_propagation(
    H::OpSum,
    system_sites::AbstractVector{<:Index},
    tester_sites::AbstractVector{<:Index},
) = JointPropagation(H, Index[system_sites...], Index[tester_sites...])

"""
    joint_unitary(U::ITensor, system_sites, tester_sites)

Store a ready-made joint system-memory Hilbert-space unitary.

`U` must contain one unprimed input and one primed output index for every
supplied system and memory site.

# Examples
```julia
s_system = siteinds("Qubit", 1)
s_tester = siteinds("Qubit", 1)
U = op("CNOT", only(s_system), only(s_tester))
action = joint_unitary(U, s_system, s_tester)
```
"""
joint_unitary(
    U::ITensor,
    system_sites::AbstractVector{<:Index},
    tester_sites::AbstractVector{<:Index},
) = JointUnitary(U, Index[system_sites...], Index[tester_sites...])

function _assert_tester_site_compatible(
    owner::AbstractString,
    action_site::Index,
    target_site::Index,
)
    _tester_site_signature(action_site) == _tester_site_signature(target_site) || throw(
        ArgumentError(
            "$owner: action site $(action_site) is incompatible with target site $(target_site).",
        ),
    )
    return nothing
end

"""
    validate_tester_action(action, memory; system_sites=Index[])

Validate an action against tester memory and, for joint actions, runtime system sites.

Compatibility is based on physical dimension and SiteType, not exact
process-tensor Index identity.
"""
validate_tester_action(::TesterIdentity, ::TesterMemory; system_sites=Index[]) = nothing

function validate_tester_action(
    action::Union{TesterPropagation,TesterUnitary},
    memory::TesterMemory;
    system_sites=Index[],
)
    isempty(system_sites) || throw(
        ArgumentError("validate_tester_action: memory-only actions do not accept system sites."),
    )
    _assert_tester_site_compatible(
        "validate_tester_action",
        only(action.sites),
        only(memory.sites),
    )
    return nothing
end

function validate_tester_action(
    action::Union{JointPropagation,JointUnitary},
    memory::TesterMemory;
    system_sites::AbstractVector{<:Index}=Index[],
)
    runtime_system = _collect_tester_sites(
        "validate_tester_action system",
        system_sites,
    )
    _assert_tester_site_compatible(
        "validate_tester_action system",
        only(action.system_sites),
        only(runtime_system),
    )
    _assert_tester_site_compatible(
        "validate_tester_action memory",
        only(action.tester_sites),
        only(memory.sites),
    )
    return nothing
end

"""
    TesterSeq

Mutable schedule of actions on memory alone or jointly on system and memory.

The schedule is independent of the physical [`TesterMemory`](@ref). Interval `k ∈
1:nsteps` surrounds process-tensor core `k`; explicit entries override
`default` there. `initialization` acts on the tester memory before interval `1`
and must be a memory-only action.
"""
mutable struct TesterSeq
    default::AbstractTesterAction
    initialization::AbstractTesterAction
    entries::Dict{Int,AbstractTesterAction}
    nsteps::Int
end

"""
    TesterSeq(default, nsteps=0; initialization=default, entries=Dict())
    TesterSeq(; default=tester_identity(), nsteps=0, initialization=default, entries=Dict())

Construct a tester-action schedule. `entries` keys are interval indices
`k ∈ 1:nsteps`; the action before interval `1` is `initialization`.

# Examples
```julia
s = siteinds("Qubit", 1)
H = OpSum() + (0.2, "X", 1)
seq = TesterSeq(nsteps=4)
add!(seq, tester_propagation(H, s), 2)
```
"""
function TesterSeq(
    default::AbstractTesterAction,
    nsteps::Int=0;
    initialization::AbstractTesterAction=default,
    entries::AbstractDict{Int,<:AbstractTesterAction}=Dict{Int,AbstractTesterAction}(),
)
    nsteps >= 0 || throw(ArgumentError("TesterSeq: nsteps must be ≥ 0; got $nsteps."))
    collected = Dict{Int,AbstractTesterAction}(pairs(entries))
    for k in keys(collected)
        k >= 1 || throw(
            ArgumentError(
                "TesterSeq: entry interval must be ≥ 1; got $k. Use `initialization=` for the action before interval 1.",
            ),
        )
        nsteps > 0 && k > nsteps && throw(
            ArgumentError("TesterSeq: entry interval $k exceeds nsteps=$nsteps."),
        )
    end
    return TesterSeq(default, initialization, collected, nsteps)
end

function TesterSeq(;
    default::AbstractTesterAction=tester_identity(),
    nsteps::Int=0,
    initialization::AbstractTesterAction=default,
    entries::AbstractDict{Int,<:AbstractTesterAction}=Dict{Int,AbstractTesterAction}(),
)
    return TesterSeq(default, nsteps; initialization=initialization, entries=entries)
end

"""
    resolve_tester_action(seq, k)

Return the explicit tester action on interval `k ≥ 1`, or `seq.default` when no
override exists. The action before interval `1` is `seq.initialization`.
"""
function resolve_tester_action(seq::TesterSeq, k::Int)
    k >= 1 || throw(
        ArgumentError("resolve_tester_action: expected interval k ≥ 1; got $k."),
    )
    return get(seq.entries, k, seq.default)
end

"""
    add!(seq::TesterSeq, action, k)

Insert `action` on interval `k ∈ 1:nsteps`, replacing an existing explicit
entry. Set the action before interval `1` with `TesterSeq(...; initialization=)`.
"""
function add!(seq::TesterSeq, action::AbstractTesterAction, k::Int)
    k >= 1 || throw(
        ArgumentError(
            "add!: tester intervals are numbered 1:nsteps; got $k. Use `initialization=` for the action before interval 1.",
        ),
    )
    seq.nsteps > 0 && k > seq.nsteps && throw(
        ArgumentError("add!: interval $k exceeds seq.nsteps=$(seq.nsteps)."),
    )
    seq.entries[k] = action
    return seq
end

function Base.:+(seq::TesterSeq, entry::Tuple{AbstractTesterAction,Int})
    add!(seq, entry[1], entry[2])
    return seq
end

add!(::InstrumentSeq, ::TesterMemory, ::Int) = throw(
    ArgumentError("add!: TesterMemory objects belong in tester workflows, not InstrumentSeq."),
)
add!(::InstrumentSeq, ::AbstractTesterAction, ::Int) = throw(
    ArgumentError("add!: tester actions belong in TesterSeq, not InstrumentSeq."),
)

function Base.show(io::IO, memory::TesterMemory)
    println(io, "ProcessTensors.TesterMemory")
    println(io, "  sites: ", length(memory.sites))
    println(io, "  space: Liouville")
    print(io, "  rho0: ", typeof(memory.rho0))
end

Base.show(io::IO, ::MIME"text/plain", memory::TesterMemory) = show(io, memory)
Base.show(io::IO, ::TesterIdentity) = print(io, "tester_identity()")

function Base.show(io::IO, seq::TesterSeq)
    ks = sort!(collect(keys(seq.entries)))
    print(
        io,
        "TesterSeq(default=$(typeof(seq.default)), nsteps=$(seq.nsteps), " *
        "$(length(ks)) explicit entries)",
    )
    print(io, "\n  initialization => ", typeof(seq.initialization))
    for k in ks
        print(io, "\n  interval $k => ", typeof(seq.entries[k]))
    end
end
