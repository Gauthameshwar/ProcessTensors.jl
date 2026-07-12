# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: src/process_tensor/build.jl
# Contributor: Gauthameshwar S.
#
# Provides the public build_process_tensor facade and dispatches to
# process-tensor builder backends.

import ITensorMPS: MPO as CoreMPO
import ITensors.Ops: Exact, Trotter

function _build_process_tensor_cores(
    method::AbstractPTBuilder,
    system::AbstractSystem,
    coupling_site::Index;
    environment::Union{Nothing,AbstractBath},
    dt::Real,
    nsteps::Int,
    alg,
    sys_alg,
)
    throw(ArgumentError("build_process_tensor: process-tensor builder $(typeof(method)) is not implemented."))
end

function _build_process_tensor_cores(
    method::Dense,
    system::AbstractSystem,
    coupling_site::Index;
    environment::Union{Nothing,AbstractBath},
    dt::Real,
    nsteps::Int,
    alg,
    sys_alg,
)
    if environment === nothing
        return _build_trivial_pt_cores(
            system,
            coupling_site,
            dt,
            nsteps,
        )
    end

    bath = environment::AbstractBath
    d_joint = joint_liouville_dim(bath, coupling_site)
    if d_joint > MAX_DENSE_LIOUVILLE_DIM
        _validate_dense_liouville_budget(d_joint; context="build_process_tensor")
    end
    if length(bath.modes) == 1
        return _build_bathmode_pt_cores(
            system,
            coupling_site,
            bath.modes[1],
            dt,
            nsteps;
            bath_coupling=bath.coupling,
            alg=alg,
            sys_alg=sys_alg,
        )
    end
    return _build_multimode_pt_cores(
        system,
        coupling_site,
        bath,
        dt,
        nsteps;
        alg=alg,
        sys_alg=sys_alg,
    )
end

"""
    build_process_tensor(system, coupling_site; method=Dense(), environment=nothing,
                         dt, nsteps, alg=Exact(), sys_alg=Trotter{1}())

Build a single-coupling-site process tensor.

`coupling_site` is the Liouville-space system leg kept as the process-tensor
input/output channel. Reuse the same Liouville index objects across the system,
bath, and instruments so later contractions match by exact index identity.

`method` selects the process-tensor construction backend. The default
[`Dense`](@ref) backend builds exact joint-Liouville cores for no-bath,
single-mode, and small multimode environments.

`alg` selects how the joint bath(+coupling) slab is built. `sys_alg` selects the
*timestep sandwich order* of free-system maps around that bath core
(`Trotter{1}()` asymmetric ``Q·M(Δt)``, `Trotter{2}()` symmetric
``M(Δt/2)·Q·M(Δt/2)``). 

System propagation is always embedded in each process-tensor slab. Insert
additional unitary control maps with [`UnitaryPropagation`](@ref) rather than
building a process tensor without system propagation.

# Examples
```julia
pt = build_process_tensor(system, coupling_site; dt=0.1, nsteps=8)
pt_sym = build_process_tensor(system, coupling_site; dt=0.1, nsteps=8, sys_alg=Trotter{2}())
```
"""
function build_process_tensor(
    system::AbstractSystem,
    coupling_site::Index;
    method::AbstractPTBuilder=Dense(),
    environment::Union{Nothing,AbstractBath}=nothing,
    dt::Real,
    nsteps::Integer,
    alg=Exact(),
    sys_alg=Trotter{1}(),
)
    nsteps_int = Int(nsteps)
    nsteps_int >= 1 || throw(ArgumentError("A process tensor requires at least one timestep; got $nsteps."))
    length(system.sites) == 1 || throw(
        ArgumentError(
            "build_process_tensor currently requires a single-site system because " *
            "system propagation is always embedded in the process tensor.",
        ),
    )
    ProcessTensors._validate_sys_alg(sys_alg)

    cores = _build_process_tensor_cores(
        method,
        system,
        coupling_site;
        environment=environment,
        dt=dt,
        nsteps=nsteps_int,
        alg=alg,
        sys_alg=sys_alg,
    )

    return ProcessTensor(CoreMPO(cores), system, environment, dt, nsteps_int, coupling_site)
end

"""
    build_process_tensor(system; method=Dense(), environment=nothing,
                         dt, nsteps, alg=Exact(), sys_alg=Trotter{1}())

Build a process tensor for a single-site system by using its only Liouville
site as the coupling site.
"""
function build_process_tensor(
    system::AbstractSystem;
    method::AbstractPTBuilder=Dense(),
    environment::Union{Nothing,AbstractBath}=nothing,
    dt::Real,
    nsteps::Integer,
    alg=Exact(),
    sys_alg=Trotter{1}(),
)
    length(system.sites) == 1 || throw(
        ArgumentError(
            "build_process_tensor(system; ...) is only allowed for single-site systems. " *
            "Pass `coupling_site::Index` explicitly for multi-site systems.",
        ),
    )
    return build_process_tensor(
        system, only(system.sites);
        method=method,
        environment=environment,
        dt=dt,
        nsteps=nsteps,
        alg=alg,
        sys_alg=sys_alg,
    )
end
