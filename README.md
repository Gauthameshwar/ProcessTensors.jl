<p align="center">
    <img src="logo.svg" width="220">
</p>

<h1 align="center">ProcessTensors.jl</h1>

<p align="center">
    MPS-based process tensors for non-Markovian open quantum systems in Julia.
</p>

<p align="center">
  <a href="https://github.com/Gauthameshwar/ProcessTensors.jl/actions/workflows/CI.yml?query=branch%3Amain">
    <img src="https://github.com/Gauthameshwar/ProcessTensors.jl/actions/workflows/CI.yml/badge.svg?branch=main" alt="CI">
  </a>
  <a href="https://codecov.io/gh/Gauthameshwar/ProcessTensors.jl">
    <img src="https://codecov.io/gh/Gauthameshwar/ProcessTensors.jl/branch/main/graph/badge.svg" alt="Coverage">
  </a>
  <a href="https://github.com/JuliaTesting/Aqua.jl">
    <img src="https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg" alt="Aqua QA">
  </a>
</p>

<p align="center">
  <a href="https://Gauthameshwar.github.io/ProcessTensors.jl">
    <img src="https://img.shields.io/badge/Read%20the%20Documentation-ProcessTensors.jl%20Docs-9558B2?style=for-the-badge&logo=julia&logoColor=white" alt="Read the Documentation"/>
  </a>
</p>

`ProcessTensors.jl` is a Julia package for constructing and contracting process tensors
as matrix-product objects. It extends the familiar `ITensorMPS.jl` workflow with
Hilbert/Liouville-space wrappers, system and bath abstractions, instrument sequences,
and high-level routines for reduced dynamics and multi-time correlations.

## Features

- Hilbert- and Liouville-space `MPS`/`MPO` wrappers built on `ITensorMPS.jl`
- Spin and bosonic system definitions with Hamiltonian and jump-operator support
- Spin and bosonic bath modes with compact bath containers
- Dense Liouville construction of single- and multi-mode process tensors
- Instrument sequences: state preparation, trace-out, observables, left/right actions, open outputs
- Reduced evolution through `evolve`
- Process contraction through `evaluate_process`
- Sequential multi-time correlations through `two_time_correlation_seq`

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/Gauthameshwar/ProcessTensors.jl")
```

## Quick Start

### Defining a Process Tensor
```julia
using ITensors
using ProcessTensors

dt = 0.1
nsteps = 24

# Physical sites
sys = siteinds("S=1/2", 1)
bath = siteinds("S=1/2", 1)

sysL = liouv_sites(sys)
bathL = liouv_sites(bath)

# System Hamiltonian
Hsys = OpSum()
Hsys += 1.0, "Sx", 1
system = spin_system(sys, Hsys)

# Bath initial state and Hamiltonian
ρbath0 = to_liouville(to_dm(MPS(bath, ["Up"])); sites=bathL)

Hbath = OpSum()
Hbath += 1.0, "Sx", 1

Hcoupling = OpSum()
Hcoupling += 1.0, "Sz", 1, "Sz", 2

mode = spin_mode(bathL, Hbath, ρbath0; coupling=Hcoupling)
environment = spin_bath([mode])

pt = build_process_tensor(system, system.sites[1]; environment, dt, nsteps)
```

### Evolve the Reduced System
```julia
ρsys0 = to_dm(MPS(sys, ["Up"]))
trajectory = evolve(pt, ρsys0)

trajectory.times
trajectory.states_liouville
```

### Evaluate Processes
```julia
obs = OpSum()
obs += 1.0, "Sz", 1

seq = default_schedule(pt)
add!(seq, 0, StatePreparation(ρsys0))
add!(seq, nsteps, ObservableMeasurement(obs))

expectation = evaluate_process(pt, seq)
```

## Examples

Runnable scripts and longer worked examples live under `scripts/` and in the
[Examples](https://Gauthameshwar.github.io/ProcessTensors.jl/stable/examples/spin_chain_unitary/)
section of the documentation:

| Topic | Script | Docs |
| ----- | ------ | ---- |
| Single-mode spin bath | `scripts/pt_tfim_singlemode.jl` | [Single spin-bath process tensor](https://Gauthameshwar.github.io/ProcessTensors.jl/stable/examples/single_spin_bath_process_tensor/) |
| Multimode spin bath | `scripts/pt_tfim_multimode.jl` | [Multimode process tensor](https://Gauthameshwar.github.io/ProcessTensors.jl/stable/examples/multimode_process_tensor/) |
| Multi-time correlations | `scripts/pt_multitime_correlations.jl` | [Multi-time correlations](https://Gauthameshwar.github.io/ProcessTensors.jl/stable/examples/multitime_correlations/) |
| TFIM TEBD (dissipative) | `scripts/tebd_tfim.jl` | [Dissipative spin](https://Gauthameshwar.github.io/ProcessTensors.jl/stable/examples/dissipative_spin/) |
| TFIM TDVP benchmark | `scripts/tdvp_tfim.jl` | [Unitary spin chain](https://Gauthameshwar.github.io/ProcessTensors.jl/stable/examples/spin_chain_unitary/) |
