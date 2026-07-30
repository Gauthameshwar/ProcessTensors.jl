# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: src/builders/ace_process_tensor.jl
# Contributor: Gauthameshwar S.
#
# Implements sequential ACE process-tensor construction by joining single-mode
# bath process tensors and compressing memory bonds after each join.
#
# Provenance:
# Join/compression algorithm adapted from Moritz Cygorek's ACE toolkit
# (https://github.com/mcygorek/ACE; Nat. Phys. 18, 662 (2022);
# J. Chem. Phys. 161, 074111 (2024)). Rewritten for ProcessTensors.jl and
# ITensors index conventions.

import ITensors.Ops: Exact, Trotter
import ITensorMPS: MPO as CoreMPO

"""
    _ace_combine_mode!(::Trotter{1}, cores, inputs, outputs, mode, coupling_site;
                       dt, nsteps, alg=Exact())

Join one bath mode onto the accumulated process-tensor `cores` with the
first-order (same-`dt`) combination: the closed single-mode core `Q_mode(Δt)`
is applied after the accumulated core on each timestep, and the two memory
links ride along as a link pair (ACE `add_modes_firstorder`/`join_thisfirst`).
"""
function _ace_combine_mode!(
    ::Trotter{1},
    cores::Vector{ITensor},
    inputs::Vector{Index},
    outputs::Vector{Index},
    mode::AbstractBathMode,
    coupling_site::Index;
    dt::Real,
    nsteps::Int,
    alg=Exact(),
)
    mode_cores, mode_in, mode_out = _build_bathmode_cores_no_sys(
        coupling_site, mode, dt, nsteps; alg=alg, run=_NO_RUN_REPORTER,
    )
    for n in 1:nsteps
        mid = prime(outputs[n], _INTERNAL_PLEV)
        accum_core = replaceind(cores[n], outputs[n], mid)
        mode_core = replaceind(
            replaceind(mode_cores[n], mode_in[n], mid),
            mode_out[n], outputs[n],
        )
        cores[n] = accum_core * mode_core
    end
    return cores
end

"""
    _ace_combine_mode!(::Trotter{2}, cores, inputs, outputs, mode, coupling_site;
                       dt, nsteps, alg=Exact())

Join one bath mode onto the accumulated process-tensor `cores` with the
symmetric half-step combination: the mode cores are built on the `Δt/2` grid
(`2 nsteps` cores) and each accumulated core is sandwiched as
`Q_mode(Δt/2) · Q_accum(Δt) · Q_mode(Δt/2)` (ACE `add_modes`/`join_symmetric`).
The internal half-step memory link of the mode is contracted inside each core.
"""
function _ace_combine_mode!(
    ::Trotter{2},
    cores::Vector{ITensor},
    inputs::Vector{Index},
    outputs::Vector{Index},
    mode::AbstractBathMode,
    coupling_site::Index;
    dt::Real,
    nsteps::Int,
    alg=Exact(),
)
    mode_cores, mode_in, mode_out = _build_bathmode_cores_no_sys(
        coupling_site, mode, dt / 2, 2 * nsteps; alg=alg, run=_NO_RUN_REPORTER,
    )
    for n in 1:nsteps
        mid_in = prime(outputs[n], _INTERNAL_PLEV)
        mid_out = prime(outputs[n], _INTERNAL_PLEV + 1)
        half_first = replaceind(
            replaceind(mode_cores[2n - 1], mode_in[2n - 1], inputs[n]),
            mode_out[2n - 1], mid_in,
        )
        accum_core = replaceind(
            replaceind(cores[n], inputs[n], mid_in),
            outputs[n], mid_out,
        )
        half_second = replaceind(
            replaceind(mode_cores[2n], mode_in[2n], mid_out),
            mode_out[2n], outputs[n],
        )
        cores[n] = half_first * accum_core * half_second
    end
    return cores
end

function _ace_combine_mode!(
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
    truncate_pt_mpo!(cores; cutoff, maxdim)

Fuse paired memory links between adjacent process-tensor cores and compress
all bonds with SVD sweeps (the sequential ACE forward/backward sweep,
delegated to `ITensorMPS.truncate!`).
"""
function truncate_pt_mpo!(cores::Vector{ITensor}; cutoff::Real, maxdim::Integer)
    length(cores) > 1 || return cores

    # Each mode join leaves an accumulated-memory link and a new-mode link
    # between adjacent cores. Fuse that pair into one MPO bond before sweeping.
    for n in 1:(length(cores) - 1)
        shared = commoninds(cores[n], cores[n + 1])
        length(shared) > 1 || continue
        fuse = combiner(shared...; tags="PT,Link,tstep=$n")
        cores[n] *= fuse
        cores[n + 1] *= dag(fuse)
    end

    mpo = CoreMPO(cores)
    ITensorMPS.truncate!(mpo; cutoff=cutoff, maxdim=maxdim)
    for j in eachindex(cores)
        cores[j] = mpo[j]
    end
    return cores
end

# Sequential ACE driver: identity PT → join each independent mode (dispatch on
# `combine_alg`) → compress bonds → embed the free-system maps with `sys_alg`.
# Uses the parent `run` from `build_process_tensor`; inner mode cores are silent.
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
            _ace_combine_mode!(
                combine_alg, cores, inputs, outputs, mode, coupling_site;
                dt=dt, nsteps=nsteps, alg=alg,
            )
            truncate_pt_mpo!(cores; cutoff=method.cutoff, maxdim=method.maxdim)
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
