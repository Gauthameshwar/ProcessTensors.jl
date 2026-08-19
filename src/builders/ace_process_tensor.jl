# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: src/builders/ace_process_tensor.jl
# Contributor: Gauthameshwar S.
#
# Implements Automated Compression of Environments (ACE) for independent bath
# modes. The environment influence is constructed as a process-tensor MPO by
# adding one microscopic mode at a time and compressing temporal memory bonds.
#
# Provenance:
# Join/compression algorithm adapted from Moritz Cygorek's ACE toolkit
# (https://github.com/mcygorek/ACE; Nat. Phys. 18, 662 (2022);
# J. Chem. Phys. 161, 074111 (2024)). Rewritten for ProcessTensors.jl and
# ITensors index conventions.

import ITensors.Ops: Exact, Trotter
import ITensorMPS: MPO as CoreMPO

# ACE construction pipeline:
# Hilbert unitary influence maps → Trotter join onto the accumulated PT →
# fuse temporal memory links → relative SVD with σ₁ rescaling →
# forward then backward sweep → next mode → embed H_S.

# --------------------------------------------------------------------------
# Mode influence and Trotter joining
# --------------------------------------------------------------------------

# Contract one mode influence into a PT timestep; dispatch selects the full-step or symmetric half-step composition.
function _ace_join_core!(
    ::Trotter{1},
    cores::Vector{ITensor},
    inputs::Vector{Index},
    outputs::Vector{Index},
    mode_cores::Vector{ITensor},
    mode_inputs::Vector{Index},
    mode_outputs::Vector{Index},
    n::Int,
)
    mid = prime(outputs[n], _INTERNAL_PLEV)
    accum_core = replaceind(cores[n], outputs[n], mid)
    mode_core = replaceind(
        replaceind(mode_cores[n], mode_inputs[n], mid),
        mode_outputs[n], outputs[n],
    )
    cores[n] = accum_core * mode_core
    return cores
end

function _ace_join_core!(
    ::Trotter{2},
    cores::Vector{ITensor},
    inputs::Vector{Index},
    outputs::Vector{Index},
    mode_cores::Vector{ITensor},
    mode_inputs::Vector{Index},
    mode_outputs::Vector{Index},
    n::Int,
)
    # Distinct temporary prime levels so the two half-step maps contract through the accumulated core without colliding with the physical PT legs.
    mid_in = prime(outputs[n], _INTERNAL_PLEV)
    mid_out = prime(outputs[n], _INTERNAL_PLEV + 1)
    half_first = replaceind(
        replaceind(mode_cores[2n - 1], mode_inputs[2n - 1], inputs[n]),
        mode_outputs[2n - 1], mid_in,
    )
    accum_core = replaceind(
        replaceind(cores[n], inputs[n], mid_in),
        outputs[n], mid_out,
    )
    half_second = replaceind(
        replaceind(mode_cores[2n], mode_inputs[2n], mid_out),
        mode_outputs[2n], outputs[n],
    )
    cores[n] = half_first * accum_core * half_second
    return cores
end

# Build closed system-mode influence maps through Hilbert-space unitaries; Trotter{2} uses twice as many half-step cores.
function _ace_mode_cores(
    ::Trotter{1},
    coupling_site::Index,
    mode::AbstractBathMode,
    dt::Real,
    nsteps::Int,
    alg,
)
    return _build_bathmode_cores_no_sys(
        coupling_site, mode, dt, nsteps;
        alg=alg, channel=:hilbert_unitary, run=_NO_RUN_REPORTER,
    )
end

function _ace_mode_cores(
    ::Trotter{2},
    coupling_site::Index,
    mode::AbstractBathMode,
    dt::Real,
    nsteps::Int,
    alg,
)
    return _build_bathmode_cores_no_sys(
        coupling_site, mode, dt / 2, 2 * nsteps;
        alg=alg, channel=:hilbert_unitary, run=_NO_RUN_REPORTER,
    )
end

# --------------------------------------------------------------------------
# Temporal-memory compression
# --------------------------------------------------------------------------

# Apply ACE's relative singular-value criterion σᵢ > ε σ₁, with `maxdim` acting only as an upper safety bound.
function _ace_relative_keep(
    singular_values::AbstractVector{<:Real};
    epsilon::Real,
    maxdim::Integer,
)
    isempty(singular_values) && return 0
    σ1 = first(singular_values)
    σ1 > 0 || return 1
    nkeep = count(σ -> σ > epsilon * σ1, singular_values)
    return clamp(nkeep, 1, min(length(singular_values), maxdim))
end

# A mode join can leave several independent memory links between neighboring time cores; fuse them before identifying or truncating the memory basis.
function _ace_fuse_bond!(cores::Vector{ITensor}, left::Int)
    shared = commoninds(cores[left], cores[left + 1])
    isempty(shared) && throw(
        ArgumentError("ACE compression: cores $left and $(left + 1) share no memory link."),
    )
    length(shared) == 1 && return only(shared)

    fuse = combiner(shared...; tags="PT,Link,tstep=$left")
    cores[left] *= fuse
    cores[left + 1] *= dag(fuse)
    return combinedind(fuse)
end

"""
    _ace_svd_bond!(cores, source, destination; epsilon, maxdim)

Compress one temporal-memory bond in the direction `source → destination`.

Any multiple shared memory links are first fused into a single bond. ACE then
retains singular directions satisfying `σᵢ > epsilon * σ₁`, subject to
`maxdim`. The source tensor stores `σ₁ U`, while the normalized remainder
`(Σ / σ₁) V` is contracted into the neighboring destination tensor.

The explicit `σ₁` rescaling keeps tensor norms well conditioned during repeated
mode joins without changing the represented process apart from the requested
truncations.
"""
function _ace_svd_bond!(
    cores::Vector{ITensor},
    source::Int,
    destination::Int;
    epsilon::Real,
    maxdim::Integer,
)
    abs(source - destination) == 1 || throw(
        ArgumentError("ACE compression requires neighboring cores; got $source and $destination."),
    )
    left = min(source, destination)
    bond = _ace_fuse_bond!(cores, left)
    source_unique = uniqueinds(cores[source], cores[destination])

    _, _, _, spectrum = svd(cores[source], source_unique; cutoff=0.0)
    singular_values = sqrt.(max.(spectrum.eigs, zero(eltype(spectrum.eigs))))
    nkeep = _ace_relative_keep(singular_values; epsilon=epsilon, maxdim=maxdim)

    U, S, V = svd(
        cores[source],
        source_unique;
        cutoff=0.0,
        maxdim=nkeep,
        lefttags=tags(bond),
    )
    σ1 = first(singular_values)
    σ1 > 0 || throw(ArgumentError("ACE compression encountered a zero tensor at core $source."))
    cores[source] = σ1 * U
    remainder = (S / σ1) * V
    cores[destination] = source < destination ?
        remainder * cores[destination] :
        cores[destination] * remainder
    return cores
end

# Recompress right-to-left after the forward join sweep so the retained memory basis incorporates information from both temporal boundaries.
function _ace_backward_sweep!(
    cores::Vector{ITensor};
    epsilon::Real,
    maxdim::Integer,
)
    for source in length(cores):-1:2
        _ace_svd_bond!(
            cores,
            source,
            source - 1;
            epsilon=epsilon,
            maxdim=maxdim,
        )
    end
    return cores
end

# --------------------------------------------------------------------------
# Sequential ACE construction
# --------------------------------------------------------------------------

"""
    _ace_join_and_sweep!(
        combine_alg, cores, inputs, outputs, mode, coupling_site;
        dt, nsteps, alg, epsilon, maxdim,
    )

Absorb one independent environment mode into an accumulated ACE process tensor.

The mode influence is constructed on the PT time grid according to
`combine_alg`. `Trotter{1}` joins one full-step mode map at each time, while
`Trotter{2}` sandwiches the accumulated PT core between two half-step mode
maps.

During the forward pass, timestep `n + 1` is joined before bond `n` is
compressed so that every memory factor generated by the new mode crosses the
bond being factorized. After the mode has been joined across the complete time
chain, a backward compression sweep propagates the resulting memory basis from
the final-time boundary back through the MPO.

Returns the mutated PT core vector.
"""
function _ace_join_and_sweep!(
    combine_alg::Union{Trotter{1},Trotter{2}},
    cores::Vector{ITensor},
    inputs::Vector{Index},
    outputs::Vector{Index},
    mode::AbstractBathMode,
    coupling_site::Index;
    dt::Real,
    nsteps::Int,
    alg=Exact(),
    epsilon::Real,
    maxdim::Integer,
)
    mode_cores, mode_inputs, mode_outputs = _ace_mode_cores(
        combine_alg, coupling_site, mode, dt, nsteps, alg,
    )

    _ace_join_core!(
        combine_alg, cores, inputs, outputs,
        mode_cores, mode_inputs, mode_outputs, 1,
    )
    for n in 1:(nsteps - 1)
        # Join one site ahead so both memory factors cross bond n before its SVD.
        _ace_join_core!(
            combine_alg, cores, inputs, outputs,
            mode_cores, mode_inputs, mode_outputs, n + 1,
        )
        _ace_svd_bond!(cores, n, n + 1; epsilon=epsilon, maxdim=maxdim)
    end
    return _ace_backward_sweep!(cores; epsilon=epsilon, maxdim=maxdim)
end

function _ace_join_and_sweep!(
    combine_alg,
    ::Vector{ITensor},
    ::Vector{Index},
    ::Vector{Index},
    ::AbstractBathMode,
    ::Index;
    kwargs...,
)
    throw(
        ArgumentError(
            "build_process_tensor: ACE combine_alg must be Trotter{1}() (same-Δt join) " *
            "or Trotter{2}() (symmetric half-Δt join); got $(typeof(combine_alg)).",
        ),
    )
end

"""
    _build_ace_pt_cores(
        method::ACE, system, coupling_site, bath;
        dt, nsteps, alg=Exact(), sys_alg=Trotter{1}(), combine_alg=Trotter{1}(),
        run=_NO_RUN_REPORTER,
    )

Construct the Liouville-space process-tensor MPO using sequential ACE compression.

The builder starts from the identity process tensor, incorporates each
independent bath mode through `_ace_join_and_sweep!`, and compresses the
temporal memory after every mode addition. Once the complete environmental
influence has been assembled, the free-system propagator is embedded on each
process-tensor timestep.

ACE requires the environment modes to be mutually independent:
bath-level couplings must therefore be empty and each system-mode interaction
must be stored on the corresponding bath mode.

Returns the process-tensor core tensors in time order.
"""
function _build_ace_pt_cores(
    method::ACE,
    system::AbstractSystem,
    coupling_site::Index,
    bath::AbstractBath;
    dt::Real,
    nsteps::Int,
    alg=Exact(),
    sys_alg=Trotter{1}(),
    combine_alg=Trotter{1}(),
    run::_AbstractRunReporter=_NO_RUN_REPORTER,
)
    _validate_sys_alg(sys_alg)
    isempty(ITensors.terms(bath.coupling)) || throw(
        ArgumentError(
            "build_process_tensor: the ACE builder requires independent bath modes, so " *
            "environment.coupling must be empty. Put each system-mode coupling on the " *
            "corresponding mode's `coupling` field.",
        ),
    )

    nmodes = length(bath.modes)
    @progress_stage run "Preparing trivial process tensor" (
        nmodes=nmodes,
        combine_alg=_info_text(string(typeof(combine_alg))),
    )
    cores = ITensor[]
    inputs = Index[]
    outputs = Index[]
    for k in 0:(nsteps - 1)
        in_k, out_k = _generate_pt_legs(coupling_site, k)
        push!(inputs, in_k)
        push!(outputs, out_k)
        push!(cores, delta(in_k, out_k))
    end

    @progress_bar run "Joining and compressing bath modes" nmodes begin
        for (k, mode) in enumerate(bath.modes)
            _ace_join_and_sweep!(
                combine_alg, cores, inputs, outputs, mode, coupling_site;
                dt=dt,
                nsteps=nsteps,
                alg=alg,
                epsilon=method.cutoff,
                maxdim=method.maxdim,
            )
            χ = ITensorMPS.maxlinkdim(CoreMPO(cores))
            @progress_update run k (mode=k, maxlinkdim=χ)
        end
    end

    χ = ITensorMPS.maxlinkdim(CoreMPO(cores))
    @progress_stage run "Embedding free-system maps" (
        sys_alg=_info_text(string(typeof(sys_alg))),
        maxlinkdim=χ,
    )
    for n in 1:nsteps
        cores[n] = _embed_system_map(cores[n], system, inputs[n], outputs[n], dt, sys_alg)
    end
    return cores
end
