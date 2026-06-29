```@meta
CurrentModule = ProcessTensors
```

# ProcessTensors.jl


`ProcessTensors.jl` is an MPS-based open quantum dynamics, Liouville-space simulation, and process tensors in Julia. It extends the familiar [`ITensorMPS.jl`](https://github.com/ITensor/ITensorMPS.jl) workflow toward density-matrix dynamics, Liouville-space superoperators, non-Markovian process tensors, and multi-time quantum observables.

It is designed for users who want to simulate open quantum systems while keeping the language of tensor networks close to the physics: states, operators, baths, instruments, observables, memory, and reduced dynamics.

---

## Why ProcessTensors.jl?

Many quantum-dynamics workflows begin in Hilbert space:

```math
|\psi(t)\rangle = U(t)|\psi(0)\rangle,
\qquad
\langle O(t)\rangle = \langle \psi(t)|O|\psi(t)\rangle.
```

But open quantum systems naturally ask for density matrices, environments, dissipation, and memory:

```math
\rho_S(t) = \operatorname{Tr}_B\!\left[
U(t)\rho_{SB}(0)U^\dagger(t)
\right].
```

`ProcessTensors.jl` provides tools for moving between these worlds.

```text
Hilbert-space dynamics
        │
        ▼
Density matrices and reduced states
        │
        ▼
Liouville-space vectorization
        │
        ▼
Liouvillian MPOs and dissipative dynamics
        │
        ▼
Process tensors with memory and instruments
        │
        ▼
Reduced evolution, observables, and multi-time correlations
```

The goal is not only to evolve a state, but to ask physically meaningful questions such as:

* How does a reduced system evolve under a structured environment?
* How do interventions at earlier times influence future outcomes?
* How can density-matrix and Liouville-space calculations be represented as MPS/MPO contractions?
* How can we evaluate observables, reduced trajectories, and multi-time correlations from the same process object?

---

## Core features

| Feature                        | What it gives you                                                                         |
| ------------------------------ | ----------------------------------------------------------------------------------------- |
| Hilbert-space MPS/MPO wrappers | Work with tensor-network states and operators in a familiar ITensorMPS style.             |
| Density-matrix construction    | Convert pure states and mixtures into density-matrix tensor-network objects.              |
| Liouville-space tools          | Vectorize density matrices and build superoperators as MPOs.                              |
| Liouvillian dynamics           | Represent Hamiltonian and dissipative dynamics in operator space.                         |
| System and bath abstractions   | Define spin/bosonic systems, bath modes, and environments.                                |
| Process tensors                | Build matrix-product representations of non-Markovian reduced dynamics.                   |
| Instruments                    | Insert state preparations, observables, trace-outs, left/right actions, and open outputs. |
| Process evaluation             | Contract process tensors with instrument sequences using `evaluate_process`.              |
| Reduced evolution              | Use `evolve` to obtain reduced system trajectories.                                       |
| Multi-time correlations        | Compute sequential and operator-insertion-style correlations.                             |

!!! note "Two pillars of the package"
    The package is built around two complementary ideas.

    First, it supports **Liouville-space tensor networks**, where density matrices are vectorized and superoperators become MPOs.

    Second, it supports **process tensors**, where the reduced dynamics of an open quantum system is stored as a multi-time tensor network that can be contracted with instruments.

---


## Quick start

The following example builds a simple process tensor for one spin coupled to one spin bath mode.

```julia
using ITensors
using ProcessTensors

dt = 0.1
nsteps = 24

# Physical Hilbert-space sites
sys = siteinds("S=1/2", 1)
bath = siteinds("S=1/2", 1)

# Liouville-space sites
sysL = liouv_sites(sys)
bathL = liouv_sites(bath)

# System Hamiltonian
Hsys = OpSum()
Hsys += 1.0, "Sx", 1
system = spin_system(sys, Hsys)

# Bath initial state
ρbath0 = to_liouville(to_dm(MPS(bath, ["Up"])); sites=bathL)

# Bath Hamiltonian
Hbath = OpSum()
Hbath += 1.0, "Sx", 1

# System-bath coupling
Hcoupling = OpSum()
Hcoupling += 1.0, "Sz", 1, "Sz", 2

mode = spin_mode(bathL, Hbath, ρbath0; coupling = Hcoupling)
environment = spin_bath([mode])

pt = build_process_tensor(system, system.sites[1]; environment, dt, nsteps)
```

The process tensor can now be used to evolve a reduced system state.

```julia
ρsys0 = to_dm(MPS(sys, ["Up"]))

trajectory = evolve(pt, ρsys0)
```

Or it can be contracted with an explicit sequence of instruments.

```julia
obs = OpSum()
obs += 1.0, "Sz", 1

seq = default_schedule(pt)
add!(seq, 0, StatePreparation(ρsys0))
add!(seq, nsteps, ObservableMeasurement(obs))

expectation = evaluate_process(pt, seq)
```

---

## Where should I start?

If you are new to the package, the recommended path is:

```text
Getting Started
  → ITensor Basics
  → MPS and MPO Basics
  → Tensor Networks in Physics
  → Density Matrices
  → Liouville Space
  → Unitary and Dissipative Dynamics
  → Single-Mode Process Tensor
```

| Goal                                          | Recommended page                                                                |
| --------------------------------------------- | ------------------------------------------------------------------------------- |
| I want to learn the basic package workflow.   | [Getting Started](getting_started.md)                                           |
| I want to understand the physics conventions. | [Theory: Hilbert and Liouville Space](theory/liouville_space.md)                |
| I want to learn ITensors index and contraction syntax. | [Tutorial: ITensor Basics](tutorials/itensor_basics.md) |
| I want to learn Hilbert-space MPS/MPO basics.        | [Tutorial: MPS and MPO Basics](tutorials/mps_mpo_basics.md)                     |
| I want to build Liouvillian MPOs.             | [Tutorial: Liouville Basics](tutorials/liouville_basics.md)                     |
| I want to simulate unitary dynamics.          | [Tutorial: Unitary Dynamics](tutorials/unitary_dynamics.md)                     |
| I want to see dissipative dynamics.                  | [Tutorial: Dissipative Dynamics](tutorials/dissipative_dynamics.md)             |
| I want non-Markovian dynamics (one bath mode).| [Tutorial: Single-Mode Process Tensor](tutorials/process_tensor_singlemode.md)  |
| I want time-dependent driving or kicked chains.      | [Examples: Driven two-level system](examples/driven_two_level_system.md)        |
| I want multimode baths or multi-time correlations. | [Examples](examples/multimode_process_tensor.md)                                |
| I want runnable end-to-end workflows.         | [Examples](examples/spin_chain_unitary.md)                                      |
| I want the function reference.                | [API Reference](api.md)                                          |

---

## Current package scope

`ProcessTensors.jl` is built for tensor-network studies of open quantum dynamics. It works best when the relevant state, operator, or process has a compact MPS/MPO representation.

| Area | Scope assessment |
|---|---|
| **Hilbert-space MPS/MPO dynamics** | Good for standard 1D tensor-network workflows, especially before entanglement growth becomes too large. |
| **Density matrices and Liouville space** | One of the main strengths of the package. Useful for vectorized density matrices, superoperators, Liouvillian MPOs, and dissipative dynamics. |
| **Unitary and dissipative evolution** | Good for short-to-intermediate time simulations. Strong quenches, chaotic dynamics, or large operator-space entanglement can become expensive. |
| **Spin and bosonic examples** | Natural for spin systems and bosonic modes with controlled level cutoffs. |
| **Process tensors** | Best suited to single-spin system process tensors with small or few-mode baths. |
| **Large baths and long memory** | Challenging. Process-tensor memory bonds can grow quickly with bath size, final time, and non-Markovian memory strength. |
| **Multi-time correlations** | A natural use case. Instruments allow measurements, trace-outs, open outputs, and operator insertions to be expressed in one workflow. |

!!! warning "Computational walls"
    The main MPS limitation is entanglement growth.

    The main process-tensor limitation is memory-bond growth.

    The package is most effective before either of these becomes too large.

---

## Citing and contributing

`ProcessTensors.jl` is under active development. Contributions, issues, examples, and discussions are welcome.

If you use the package in research, please cite the package repository and any relevant process-tensor or tensor-network literature listed in the [References](references.md) page.
