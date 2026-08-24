# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: src/instruments/tester_compile.jl
# Contributor: Gauthameshwar S.
#
# Materializes tester actions and compiles their persistent memory wire into
# process-tensor instrument tensors.

const _TesterOnlyAction = Union{TesterIdentity,TesterPropagation,TesterUnitary}
const _JointTesterAction = Union{JointPropagation,JointUnitary}

# Recover Hilbert sites from mixed Hilbert/Liouville tester indices.
function _physical_tester_sites(sites::AbstractVector{<:Index})
    return Index[
        has_tag_token(site, "Liouv") ? _phys_site_from_liouv(site) : site
        for site in sites
    ]
end

# Retarget a canonical Liouville map without changing its vectorization order.
function _retarget_liouville_map(
    map::ITensor,
    canonical_sites::AbstractVector{<:Index},
    input_sites::AbstractVector{<:Index},
    output_sites::AbstractVector{<:Index},
)
    length(canonical_sites) == length(input_sites) == length(output_sites) || throw(
        ArgumentError("_retarget_liouville_map: site count mismatch."),
    )
    result = map
    for (site, target) in zip(canonical_sites, output_sites)
        result = replaceind(result, site, target)
    end
    for (site, target) in zip(canonical_sites, input_sites)
        result = replaceind(result, prime(site), target)
    end
    return result
end

# Build exp(-i H dt) as a Liouville map and attach it to the runtime legs.
function _opsum_liouville_map(
    H::OpSum,
    action_sites::AbstractVector{<:Index},
    input_sites::AbstractVector{<:Index},
    output_sites::AbstractVector{<:Index},
    dt::Real,
)
    physical_sites = _physical_tester_sites(action_sites)
    canonical_sites = liouv_sites(physical_sites)
    map = ProcessTensors._hilbert_unitary_liouville_propagator(
        H,
        canonical_sites,
        dt,
    )
    return _retarget_liouville_map(
        map,
        canonical_sites,
        input_sites,
        output_sites,
    )
end

# Dense unitary matrix from a Hilbert ITensor; reject non-unitary data.
function _unitary_matrix(
    U::ITensor,
    physical_sites::AbstractVector{<:Index};
    owner::AbstractString,
)
    d = prod(dim.(physical_sites))
    matrix = reshape(
        ComplexF64.(Array(U, prime.(physical_sites)..., physical_sites...)),
        d,
        d,
    )
    identity_matrix = Matrix{ComplexF64}(I, d, d)
    isapprox(matrix' * matrix, identity_matrix; atol=1e-10, rtol=1e-10) || throw(
        ArgumentError("$owner: supplied ITensor is not unitary within tolerance."),
    )
    return matrix
end

# Convert a Hilbert unitary into a Liouville superoperator on the runtime legs.
function _unitary_liouville_map(
    U::ITensor,
    physical_sites::AbstractVector{<:Index},
    input_sites::AbstractVector{<:Index},
    output_sites::AbstractVector{<:Index};
    owner::AbstractString,
)
    _unitary_matrix(U, physical_sites; owner)
    canonical_sites = liouv_sites(physical_sites)
    map = _hilbert_itensor_to_liouville(U, physical_sites, canonical_sites)
    return _retarget_liouville_map(
        map,
        canonical_sites,
        input_sites,
        output_sites,
    )
end

# Use the principal Schur phases so the half map is unitary and squares to `U`.
function _unitary_half(
    U::ITensor,
    physical_sites::AbstractVector{<:Index},
)
    matrix = _unitary_matrix(U, physical_sites; owner="JointUnitary")
    factorization = schur(matrix)
    phases = angle.(diag(factorization.T))
    half_matrix = factorization.Z * Diagonal(exp.(0.5im .* phases)) * factorization.Z'
    identity_matrix = Matrix{ComplexF64}(I, size(matrix, 1), size(matrix, 2))
    isapprox(half_matrix * half_matrix, matrix; atol=1e-9, rtol=1e-9) || throw(
        ArgumentError("JointUnitary: could not construct a square root of the supplied unitary."),
    )
    isapprox(half_matrix' * half_matrix, identity_matrix; atol=1e-10, rtol=1e-10) || throw(
        ArgumentError("JointUnitary: constructed square root is not unitary."),
    )
    dims = Tuple(vcat(dim.(physical_sites), dim.(physical_sites)))
    return ITensor(
        reshape(half_matrix, dims),
        prime.(physical_sites)...,
        physical_sites...,
    )
end

# Identity tester action: copy memory from input to output.
function _materialize_tester_action(
    ::TesterIdentity,
    memory_input::Index,
    memory_output::Index,
    ::Real,
)
    return delta(memory_input, memory_output)
end

# Tester-only Hamiltonian step on the memory wire.
function _materialize_tester_action(
    action::TesterPropagation,
    memory_input::Index,
    memory_output::Index,
    dt::Real,
)
    return _opsum_liouville_map(
        action.H,
        action.sites,
        Index[memory_input],
        Index[memory_output],
        dt,
    )
end

# Tester-only unitary on the memory wire.
function _materialize_tester_action(
    action::TesterUnitary,
    memory_input::Index,
    memory_output::Index,
    ::Real,
)
    return _unitary_liouville_map(
        action.U,
        action.sites,
        Index[memory_input],
        Index[memory_output];
        owner="TesterUnitary",
    )
end

# Joint Hamiltonian map on system ⊗ tester for duration `dt`.
function _materialize_tester_action(
    action::JointPropagation,
    system_input::Index,
    system_output::Index,
    memory_input::Index,
    memory_output::Index,
    dt::Real,
)
    return _opsum_liouville_map(
        action.H,
        vcat(action.system_sites, action.tester_sites),
        Index[system_input, memory_input],
        Index[system_output, memory_output],
        dt,
    )
end

# Split a joint Hamiltonian into G^{1/2} before and after the PT core.
function _materialize_joint_halves(
    action::JointPropagation,
    pre_system::Index,
    core_input::Index,
    core_output::Index,
    post_system::Index,
    memory_input::Index,
    memory_mid::Index,
    memory_output::Index,
    half_dt::Real,
)
    pre = _materialize_tester_action(
        action,
        pre_system,
        core_input,
        memory_input,
        memory_mid,
        half_dt,
    )
    post = _materialize_tester_action(
        action,
        core_output,
        post_system,
        memory_mid,
        memory_output,
        half_dt,
    )
    return pre, post
end

# Split a joint unitary into √U before and after the PT core.
function _materialize_joint_halves(
    action::JointUnitary,
    pre_system::Index,
    core_input::Index,
    core_output::Index,
    post_system::Index,
    memory_input::Index,
    memory_mid::Index,
    memory_output::Index,
    ::Real,
)
    physical_sites = vcat(action.system_sites, action.tester_sites)
    half_unitary = _unitary_half(action.U, physical_sites)
    pre = _unitary_liouville_map(
        half_unitary,
        physical_sites,
        Index[pre_system, memory_input],
        Index[core_input, memory_mid];
        owner="JointUnitary",
    )
    post = _unitary_liouville_map(
        half_unitary,
        physical_sites,
        Index[core_output, memory_mid],
        Index[post_system, memory_output];
        owner="JointUnitary",
    )
    return pre, post
end

# Fresh system index with the same tags/prime as a PT leg, used to insert G^{1/2}.
function _temporary_system_leg(site::Index)
    temporary = Index(dim(site); tags=tags(site))
    return plev(site) == 0 ? temporary : prime(temporary, plev(site))
end

# Tester-only interval: full map in `pre`, identity `post`.
function _materialize_interval(
    action::_TesterOnlyAction,
    ::Index,
    ::Index,
    memory_input::Index,
    memory_output::Index,
    dt::Real,
)
    return (
        pre=_materialize_tester_action(action, memory_input, memory_output, dt),
        post=ITensor(1.0),
        pre_system=nothing,
        post_system=nothing,
    )
end

# Joint interval: G^{1/2} on the input (`pre`) and G^{1/2} on the output (`post`).
function _materialize_interval(
    action::_JointTesterAction,
    core_input::Index,
    core_output::Index,
    memory_input::Index,
    memory_output::Index,
    dt::Real,
)
    memory_mid = Index(dim(memory_input), "Tester,Memory,Mid")
    pre_system = _temporary_system_leg(core_input)
    post_system = _temporary_system_leg(core_output)
    half_dt = dt / 2
    pre, post = _materialize_joint_halves(
        action,
        pre_system,
        core_input,
        core_output,
        post_system,
        memory_input,
        memory_mid,
        memory_output,
        half_dt,
    )
    return (
        pre=pre,
        post=post,
        pre_system=pre_system,
        post_system=post_system,
    )
end

# Bind the tstep=0 tester-only map from stored tester sites onto the first memory link.
function _initial_tester_map(
    action::_TesterOnlyAction,
    memory::Tester,
    memory_output::Index,
    dt::Real,
    ::AbstractString,
)
    validate_tester_action(action, memory)
    return _materialize_tester_action(
        action,
        only(memory.sites),
        memory_output,
        dt,
    )
end

# Reject joint actions at tstep=0; they have no preceding system output.
function _initial_tester_map(
    ::_JointTesterAction,
    ::Tester,
    ::Index,
    ::Real,
    owner::AbstractString,
)
    throw(ArgumentError("$owner: joint tester actions are not allowed at tstep=0."))
end

# Replace `source` by `target` only when the tensor still carries `source`.
_relabel_if_present(tensor::ITensor, source::Index, target::Index) =
    hasind(tensor, source) ? replaceind(tensor, source, target) : tensor

# Check a tester-only action against stored tester sites.
_validate_runtime_action(action::_TesterOnlyAction, memory::Tester, ::AbstractVector{<:Index}) =
    validate_tester_action(action, memory)

# Check a joint action against both system sites and stored tester sites.
_validate_runtime_action(
    action::_JointTesterAction,
    memory::Tester,
    system_sites::AbstractVector{<:Index},
) = validate_tester_action(action, memory; system_sites)

# Fold tester init + interval pres onto instruments; keep pending posts for evolve snapshots.
function _compile_tester_control_data(
    pt,
    instruments::AbstractVector{<:ITensor},
    memory::Tester,
    schedule::TesterSeq,
    owner::AbstractString,
)
    schedule.nsteps == pt.nsteps || throw(
        ArgumentError(
            "$owner: tester_seq.nsteps=$(schedule.nsteps) must equal " *
            "pt.nsteps=$(pt.nsteps).",
        ),
    )
    length(instruments) == pt.nsteps + 1 || throw(
        ArgumentError("$owner: instrument tensor count does not match pt.nsteps."),
    )

    system_sites = Index[_phys_site_from_liouv(site) for site in pt.system.sites]
    memory_dim = dim(only(memory.sites))
    memory_links = Index[
        Index(memory_dim, "Tester,Memory,tstep=$k")
        for k in 0:pt.nsteps
    ]

    initial_action = resolve_tester_action(schedule, 0)
    initial_map = _initial_tester_map(
        initial_action,
        memory,
        memory_links[1],
        pt.dt,
        owner,
    )
    initial_state = _coerce_liouville_state(memory.rho0, memory.sites)
    initial_tensor = _reindex_itensor(
        _mps_to_itensor(initial_state),
        siteinds(initial_state),
        memory.sites,
    )

    intervals = map(1:pt.nsteps) do k
        action = resolve_tester_action(schedule, k)
        _validate_runtime_action(action, memory, system_sites)
        core_input = only(ProcessTensors.input_sites(pt, k - 1))
        core_output = only(ProcessTensors.output_sites(pt, k - 1))
        return _materialize_interval(
            action,
            core_input,
            core_output,
            memory_links[k],
            memory_links[k + 1],
            pt.dt,
        )
    end

    compiled = ITensor[instruments...]
    first_interval = first(intervals)
    first_input = only(ProcessTensors.input_sites(pt, 0))
    initial = compiled[1] * initial_tensor * initial_map
    if first_interval.pre_system !== nothing
        initial = _relabel_if_present(
            initial,
            first_input,
            first_interval.pre_system,
        )
    end
    compiled[1] = initial * first_interval.pre

    for step in 1:(pt.nsteps - 1)
        previous = intervals[step]
        following = intervals[step + 1]
        output_previous, input_current = ProcessTensors.coupling_times(pt, step)
        output_leg = only(output_previous)
        input_leg = only(input_current)
        bridge = compiled[step + 1]

        if previous.post_system !== nothing
            bridge = _relabel_if_present(
                bridge,
                output_leg,
                previous.post_system,
            )
        end
        bridge *= previous.post

        if following.pre_system !== nothing
            bridge = _relabel_if_present(
                bridge,
                input_leg,
                following.pre_system,
            )
        end
        compiled[step + 1] = bridge * following.pre
    end

    return (
        instruments=compiled,
        pending_post=ITensor[interval.post for interval in intervals],
        post_system=Union{Nothing,Index}[interval.post_system for interval in intervals],
        memory_output=memory_links[2:end],
        tester_sites=memory.sites,
    )
end

# Evaluate compile: apply the last post-half and trace the tester memory wire.
function _compile_tester_control(
    pt,
    instruments::AbstractVector{<:ITensor},
    memory::Tester,
    schedule::TesterSeq,
)
    data = _compile_tester_control_data(
        pt,
        instruments,
        memory,
        schedule,
        "evaluate_process",
    )
    compiled = data.instruments
    final_output = only(ProcessTensors.output_sites(pt, pt.nsteps - 1))
    terminal = compiled[end]
    final_post_system = last(data.post_system)
    if final_post_system !== nothing
        terminal = _relabel_if_present(
            terminal,
            final_output,
            final_post_system,
        )
    end
    terminal *= last(data.pending_post)
    terminal *= _vectorized_identity_itensor(Index[last(data.memory_output)])
    compiled[end] = terminal
    return compiled
end

# Missing tester_seq: compile identity tester dynamics.
function _compile_tester_control_data(
    pt,
    instruments::AbstractVector{<:ITensor},
    memory::Tester,
    ::Nothing,
    owner::AbstractString,
)
    schedule = TesterSeq(default=tester_identity(), nsteps=pt.nsteps)
    return _compile_tester_control_data(pt, instruments, memory, schedule, owner)
end

# No tester: leave the instrument tensors unchanged.
_compile_tester_control(pt, instruments, ::Nothing, ::Nothing) = instruments

# A schedule without memory is invalid.
function _compile_tester_control(
    ::Any,
    ::AbstractVector{<:ITensor},
    ::Nothing,
    ::TesterSeq,
)
    throw(ArgumentError("evaluate_process: tester_seq requires a tester."))
end

# Missing tester_seq: evaluate with identity tester dynamics.
function _compile_tester_control(
    pt,
    instruments::AbstractVector{<:ITensor},
    memory::Tester,
    ::Nothing,
)
    schedule = TesterSeq(default=tester_identity(), nsteps=pt.nsteps)
    return _compile_tester_control(pt, instruments, memory, schedule)
end
