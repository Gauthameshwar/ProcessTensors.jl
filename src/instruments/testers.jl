# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: src/instruments/testers.jl
# Contributor: Gauthameshwar S.
#
# Defines memory-bearing tester ancillas, tester actions, and intervention
# schedules for later process-tensor contraction.

# Return the physical local dimension and SiteType encoded by a Hilbert or
# Liouville site. Actions are compared by physical meaning, not Index identity.
function _tester_site_signature(site::Index)
    family = _liouv_site_type(tag_tokens(site))
    family === nothing && throw(
        ArgumentError("Could not infer a physical SiteType from tester site $(site)."),
    )
    if has_tag_token(site, "Liouv")
        d2 = dim(site)
        d = isqrt(d2)
        d * d == d2 || throw(
            ArgumentError("Liouville tester site $(site) has dim=$d2, expected a perfect square."),
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
    tester_sites::AbstractVector{<:Index},
)
    state_sites = Index[siteinds(state)...]
    state_sites == tester_sites || throw(
        ArgumentError(
            "Tester: Liouville rho0 must use the exact tester site indices. " *
            "Got $(state_sites) and $(tester_sites).",
        ),
    )
    return nothing
end

function _validate_tester_initial_state(
    state::Union{AbstractMPS{Hilbert},AbstractMPO{Hilbert}},
    tester_sites::AbstractVector{<:Index},
)
    state_sites = if state isa AbstractMPO{Hilbert}
        _phys_sites_from_hilbert_mpo(state)
    else
        Index[siteinds(state)...]
    end
    length(state_sites) == length(tester_sites) || throw(
        ArgumentError(
            "Tester: rho0 has $(length(state_sites)) physical sites, expected $(length(tester_sites)).",
        ),
    )
    for (state_site, tester_site) in zip(state_sites, tester_sites)
        _tester_site_signature(state_site) == _tester_site_signature(tester_site) || throw(
            ArgumentError(
                "Tester: rho0 site $(state_site) is incompatible with tester site $(tester_site).",
            ),
        )
    end
    return nothing
end

"""
    Tester(sites, rho0)

Memory-bearing ancillary quantum object used by a process-tensor tester.

`sites` may contain one Hilbert- or Liouville-space site of any physical
SiteType; Hilbert input is normalized with [`liouv_sites`](@ref). `rho0` is the
initial tester state and must match the tester site's physical dimension and
SiteType. A `Tester` stores no Hamiltonian or intervention schedule.

Use [`tester`](@ref) for ordinary construction.
"""
struct Tester{S<:Union{AbstractMPS,AbstractMPO{Hilbert}}} <: AbstractInstrument
    sites::Vector{Index}
    rho0::S

    function Tester(
        sites::AbstractVector{<:Index},
        rho0::S,
    ) where {S<:Union{AbstractMPS,AbstractMPO{Hilbert}}}
        collected = _collect_tester_sites("Tester", sites)
        # Store Liouville sites; Hilbert input is converted with `liouv_sites`.
        tester_sites = has_tag_token(only(collected), "Liouv") ? collected : liouv_sites(collected)
        _validate_tester_initial_state(rho0, tester_sites)
        return new{S}(tester_sites, rho0)
    end
end

"""
    tester(sites, rho0)

Construct a single-site [`Tester`](@ref) ancillary object.

# Examples
```julia
s = siteinds("Qubit", 1)
rho0 = to_dm(MPS(s, ["0"]))
memory = tester(s, rho0)
```
"""
tester(sites::AbstractVector{<:Index}, rho0) = Tester(sites, rho0)

"""
    AbstractTesterAction

Abstract interface for tester-only and joint system-tester actions.

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

Hamiltonian description for tester-only propagation.
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

Ready-made Hilbert-space unitary acting on the tester site.
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

Hamiltonian description for joint system-tester propagation.
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
        memory = _collect_tester_sites("JointPropagation tester", tester_sites)
        _validate_disjoint_action_sites("JointPropagation", system, memory)
        return new{T}(H, system, memory)
    end
end

"""
    JointUnitary

Ready-made Hilbert-space unitary acting jointly on system and tester sites.
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
        memory = _collect_tester_sites("JointUnitary tester", tester_sites; hilbert_only=true)
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
        throw(ArgumentError("$owner: system and tester sites must be distinct indices."))
    return nothing
end

"""
    tester_identity()

Construct the default action that carries tester memory forward unchanged.
"""
tester_identity() = TesterIdentity()

"""
    tester_propagation(H::OpSum, sites)

Store a tester-only Hamiltonian for later propagation on `sites`.

The Hamiltonian is stored as a lazy object and used when constructing the unitary during `evaluate_process`.
"""
tester_propagation(H::OpSum, sites::AbstractVector{<:Index}) =
    TesterPropagation(H, Index[sites...])

"""
    tester_unitary(U::ITensor, sites)

Store a ready-made tester-only Hilbert-space unitary.

`U` must contain each supplied site as an unprimed input index and the
corresponding primed output index.
"""
tester_unitary(U::ITensor, sites::AbstractVector{<:Index}) =
    TesterUnitary(U, Index[sites...])

"""
    joint_propagation(H::OpSum, system_sites, tester_sites)

Store a joint system-tester Hamiltonian for later propagation.

OpSum site numbers refer to `vcat(system_sites, tester_sites)`. For the initial
one-system-site/one-tester-site implementation, site `1` is the system and site
`2` is the tester. The Hamiltonian is stored as a lazy object and used when 
constructing the unitary during `evaluate_process`.
"""
joint_propagation(
    H::OpSum,
    system_sites::AbstractVector{<:Index},
    tester_sites::AbstractVector{<:Index},
) = JointPropagation(H, Index[system_sites...], Index[tester_sites...])

"""
    joint_unitary(U::ITensor, system_sites, tester_sites)

Store a ready-made joint system-tester Hilbert-space unitary.

`U` must contain one unprimed input and one primed output index for every
supplied system and tester site. 
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
    validate_tester_action(action, tester; system_sites=Index[])

Validate an action against a tester and, for joint actions, runtime system sites.

Compatibility is based on physical dimension and SiteType, not exact
process-tensor Index identity.
"""
validate_tester_action(::TesterIdentity, ::Tester; system_sites=Index[]) = nothing

function validate_tester_action(
    action::Union{TesterPropagation,TesterUnitary},
    memory::Tester;
    system_sites=Index[],
)
    isempty(system_sites) || throw(
        ArgumentError("validate_tester_action: tester-only actions do not accept system sites."),
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
    memory::Tester;
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
        "validate_tester_action tester",
        only(action.tester_sites),
        only(memory.sites),
    )
    return nothing
end

"""
    TesterSeq

Mutable schedule of tester-only and joint system-tester actions.

The schedule is independent of the physical [`Tester`](@ref). Explicit entries
override `default`, including at `tstep = 0`.
"""
mutable struct TesterSeq
    default::AbstractTesterAction
    entries::Dict{Int,AbstractTesterAction}
    nsteps::Int
end

"""
    TesterSeq(default, nsteps=0; entries=Dict())
    TesterSeq(; default=tester_identity(), nsteps=0, entries=Dict())

Construct a tester-action schedule.

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
    entries::AbstractDict{Int,<:AbstractTesterAction}=Dict{Int,AbstractTesterAction}(),
)
    nsteps >= 0 || throw(ArgumentError("TesterSeq: nsteps must be ≥ 0; got $nsteps."))
    collected = Dict{Int,AbstractTesterAction}(pairs(entries))
    for tstep in keys(collected)
        tstep >= 0 || throw(
            ArgumentError("TesterSeq: entry timestep must be ≥ 0; got $tstep."),
        )
        nsteps > 0 && tstep > nsteps && throw(
            ArgumentError("TesterSeq: entry tstep=$tstep exceeds nsteps=$nsteps."),
        )
    end
    return TesterSeq(default, collected, nsteps)
end

function TesterSeq(;
    default::AbstractTesterAction=tester_identity(),
    nsteps::Int=0,
    entries::AbstractDict{Int,<:AbstractTesterAction}=Dict{Int,AbstractTesterAction}(),
)
    return TesterSeq(default, nsteps; entries=entries)
end

"""
    resolve_tester_action(seq, k)

Return the explicit tester action at timestep `k`, or `seq.default` when no
override exists.
"""
function resolve_tester_action(seq::TesterSeq, k::Int)
    k >= 0 || throw(
        ArgumentError("resolve_tester_action: expected k ≥ 0; got $k."),
    )
    return get(seq.entries, k, seq.default)
end

"""
    add!(seq::TesterSeq, action, tstep)

Insert `action` at `tstep`, replacing an existing explicit entry.
"""
function add!(seq::TesterSeq, action::AbstractTesterAction, tstep::Int)
    tstep >= 0 || throw(ArgumentError("add!: tstep must be ≥ 0; got $tstep."))
    seq.nsteps > 0 && tstep > seq.nsteps && throw(
        ArgumentError("add!: tstep=$tstep exceeds seq.nsteps=$(seq.nsteps)."),
    )
    seq.entries[tstep] = action
    return seq
end

add!(::InstrumentSeq, ::Tester, ::Int) = throw(
    ArgumentError("add!: Tester objects belong in tester workflows, not InstrumentSeq."),
)
add!(::InstrumentSeq, ::AbstractTesterAction, ::Int) = throw(
    ArgumentError("add!: tester actions belong in TesterSeq, not InstrumentSeq."),
)

function Base.show(io::IO, memory::Tester)
    println(io, "ProcessTensors.Tester")
    println(io, "  sites: ", length(memory.sites))
    println(io, "  space: Liouville")
    print(io, "  rho0: ", typeof(memory.rho0))
end

Base.show(io::IO, ::MIME"text/plain", memory::Tester) = show(io, memory)
Base.show(io::IO, ::TesterIdentity) = print(io, "tester_identity()")

function Base.show(io::IO, seq::TesterSeq)
    ks = sort!(collect(keys(seq.entries)))
    print(
        io,
        "TesterSeq(default=$(typeof(seq.default)), nsteps=$(seq.nsteps), " *
        "$(length(ks)) explicit entries)",
    )
    for k in ks
        print(io, "\n  tstep=$k => ", typeof(seq.entries[k]))
    end
end
