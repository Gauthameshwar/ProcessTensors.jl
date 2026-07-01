# ProcessTensors.jl

`ProcessTensors.jl` is a Julia package for MPS-based open quantum dynamics, Liouville-space simulation, and process tensors.

---

## Why this project exists

Most open-quantum-softwares that exist today handles Markovian 
dynamics well, but say little about what happens when the
environment remembers. As quantum devices push
into regimes of strong coupling and structured environments, memory effects
become the rule rather than the exception, and **process tensors** have
emerged as the natural language for describing them.

The tools for working with process tensors, however, remain
sparse for a field this active. `ProcessTensors.jl` exists to 
close that gap. This package primarily uses MPS/MPO infrastructure to 
define and manipulate process tensors in a memory-efficient way. 
It's built natively on `ITensorMPS.jl`, deliberately 
separating low-level tensor-network machinery from high-level physics workflows, 
so that the codes appear as natural and close to the theory. 
Alongside the code, the documentation is written to teach the basics 
of Liouville-space and process-tensor formalism, paired with runnable 
examples that mirror the underlying equations closely.

Today, `ProcessTensors.jl` covers the essentials: single-mode and small
multimode process tensors, spin and bosonic baths, reduced dynamics, and
multi-time correlations. The next concrete milestone is extending this to
large, realistic non-Markovian environments via the ACE algorithm — a
near-term, committed goal rather than a distant aspiration.

Beyond that, the longer-term vision is for this package to grow from a
clean, accessible implementation into a genuine research platform: a place
where process-tensor and open-system tensor-network algorithms —
TEMPO, PT-TEMPO, TEDOPA, and other influence-functional-based methods — can
be implemented side by side, benchmarked against each other, taught to
newcomers, and reused by researchers who'd rather build on solid
infrastructure than rebuild it from scratch.

---

## What the package currently offers

The current package is built around two complementary pillars.

First, it supports **Liouville-space tensor networks**. Density matrices can be vectorized, and superoperators can be represented as MPOs. This makes density-matrix and dissipative dynamics feel close to the usual MPS/MPO workflow.

Second, it supports **process tensors**. A process tensor stores the reusable influence of an environment over time. Once it is built, different instrument sequences can be contracted with it to compute reduced trajectories, observables, open outputs, and multi-time quantities.

| Feature                        | What it gives you                                                                         |
| ------------------------------ | ----------------------------------------------------------------------------------------- |
| Hilbert-space MPS/MPO wrappers | Work with tensor-network states and operators in a familiar ITensorMPS style.             |
| Density-matrix construction    | Convert pure states and mixtures into density-matrix tensor-network objects.              |
| Liouville-space tools          | Vectorize density matrices and build superoperators as MPOs.                              |
| Liouvillian dynamics           | Represent Hamiltonian and dissipative dynamics in operator space.                         |
| System and bath abstractions   | Define spin/bosonic systems, bath modes, and environments.                                |
| Process tensors                | Build matrix-product representations of open-system memory.                               |
| Instruments                    | Insert state preparations, observables, trace-outs, left/right actions, and open outputs. |
| Process evaluation             | Contract process tensors with instrument sequences using `evaluate_process`.              |
| Reduced evolution              | Use `evolve` to obtain reduced system trajectories.                                       |
| Multi-time correlations        | Compute sequential and operator-insertion-style correlations.                             |

!!! tip "Time evolution, not ground-state search"
    This package builds on the **time-evolution** side of
    [`ITensorMPS.jl`](https://github.com/ITensor/ITensorMPS.jl): `tebd`, `tdvp`,
    Liouville propagators, and process-tensor contraction workflows.

    It does **not** focus on DMRG, variational ground-state search, or equilibrium
    spectral methods.

---

## Where to start

Most of what makes this package distinctive appears in two tutorials:

* **[Dissipative Dynamics](tutorials/dissipative_dynamics.md)** — open-system evolution in Liouville space: Lindblad generators, `MPO_Liouville`, TEBD/TDVP on density matrices.
* **[Single-Mode Process Tensor](tutorials/process_tensor_singlemode.md)** — process-tensor construction, bath memory, instruments, `evolve`, and `evaluate_process`.

Everything else in the documentation supports those two pages: ITensor syntax, MPS/MPO objects, Hilbert-versus-Liouville conventions, and closed-system dynamics as stepping stones.

Choose a path that suits you best:

| Background                                                           | Suggested route                                                                                                                                                                                                                                                                                                                                                                                                                          |
| -------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **You code and know the theory.**                                    | Start with [Dissipative Dynamics](tutorials/dissipative_dynamics.md), then [Single-Mode Process Tensor](tutorials/process_tensor_singlemode.md). Skim [Hilbert and Liouville Space](theory/liouville_space.md) and [Process Tensors](theory/process_tensors.md) for conventions only. Use [Examples](examples/spin_chain_unitary.md) for scripts and [API Reference](api.md) for the function list.                                      |
| **You know the theory but are new to ITensors.**                     | [ITensor Basics](tutorials/itensor_basics.md) → [MPS and MPO Basics](tutorials/mps_mpo_basics.md) → [Liouville-Space Basics](tutorials/liouville_basics.md) → the two core tutorials above. Use theory pages as a convention dictionary, not the main route.                                                                                                                                                                             |
| **You are new to open quantum systems or tensor networks in Julia.** | [Installation](installation.md) → [ITensor Basics](tutorials/itensor_basics.md) → [MPS and MPO Basics](tutorials/mps_mpo_basics.md) → [Liouville-Space Basics](tutorials/liouville_basics.md) → [Unitary Dynamics](tutorials/unitary_dynamics.md) → [Dissipative Dynamics](tutorials/dissipative_dynamics.md) → [Single-Mode Process Tensor](tutorials/process_tensor_singlemode.md). Read theory when a concept or notation is unclear. |

!!! note "Tutorials are cumulative"
    Later tutorials assume syntax and conventions from earlier ones, especially ITensor index identity, shared `sites_L`, and density-matrix vectorisation.

!!! tip "Theory pages"
    Use the theory section as a convention dictionary, not as a prerequisite course for every tutorial.

---

## Quick start

If you already know where you are headed, the snippet below builds a one-spin system coupled to one spin bath mode and constructs a process tensor. For context and checks along the way, use [Single-Mode Process Tensor](tutorials/process_tensor_singlemode.md).

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
ψmps = MPS(bath, ["Up"])
ρmpo = to_dm(ψmps)
ρbath0 = to_liouville(ρmpo; sites=bathL)

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

## Find a doc page by topic

| Goal                                        | Page                                                                 |
| ------------------------------------------- | -------------------------------------------------------------------- |
| Open-system / Lindblad evolution            | [Dissipative Dynamics](tutorials/dissipative_dynamics.md)            |
| Process tensors, baths, and instruments     | [Single-Mode Process Tensor](tutorials/process_tensor_singlemode.md) |
| Package setup                               | [Installation](installation.md)                                      |
| Physics conventions and notation            | [Theory: Hilbert and Liouville Space](theory/liouville_space.md)     |
| ITensor index and contraction syntax        | [Tutorial: ITensor Basics](tutorials/itensor_basics.md)              |
| Hilbert-space MPS/MPO                       | [Tutorial: MPS and MPO Basics](tutorials/mps_mpo_basics.md)          |
| Vectorized density matrices                 | [Tutorial: Liouville Basics](tutorials/liouville_basics.md)          |
| Closed-system TEBD/TDVP                     | [Tutorial: Unitary Dynamics](tutorials/unitary_dynamics.md)          |
| Time-dependent or kicked models             | [Examples](examples/driven_two_level_system.md)                      |
| Multimode baths and multi-time correlations | [Examples](examples/multimode_process_tensor.md)                     |
| End-to-end scripts                          | [Examples](examples/spin_chain_unitary.md)                           |
| Function reference                          | [API Reference](api.md)                                              |

!!! note "Examples"
    Example pages are model-oriented scripts and variations. Some advanced cards still point to runnable scripts while fuller Literate walkthroughs are being developed.

---

## Citing and contributing

`ProcessTensors.jl` is under active development. Contributions from expert developers, bug reports from users, new examples, and discussions about future directions are very welcome.

If you use the package in research, please cite the [ProcessTensors.jl repository](https://github.com/Gauthameshwar/ProcessTensors.jl) and any relevant process-tensor or tensor-network literature cited in the theory pages.
