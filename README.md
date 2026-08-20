<p align="center">
    <img src="logo.svg" width="220">
</p>

<h1 align="center">ProcessTensors.jl</h1>

<p align="center">
    <b>Tensor-network methods for process tensors and non-Markovian quantum dynamics.</b>
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

`ProcessTensors.jl` is a Julia package for simulating open quantum systems with matrix-product states and operators. It is built on `ITensorMPS.jl` and keeps the physics-facing workflow close to the equations: define a system, describe its environment, build a process tensor, and then ask what experiments that process would produce.

The same package can also be used before process tensors enter the story: closed-system Hilbert-space dynamics, vectorized Liouville-space evolution, Lindblad generators, TEBD, TDVP, driven systems, and dissipative many-body models all live under the same interface.

## What can you do with it?

| | |
| --- | --- |
| **Evolve quantum states** | Work with Hilbert- and Liouville-space `MPS`/`MPO` objects, unitary dynamics, Lindblad evolution, TEBD, TDVP, and time-dependent Hamiltonians. |
| **Build environmental memory** | Construct spin and bosonic baths mode by mode, then turn their influence into an MPO process tensor with `Dense()` or **ACE**. |
| **Run experiments on a process** | Assemble preparations, controls, measurements, left/right actions, trace-outs, and open legs into an `InstrumentSeq`, then contract it with `evaluate_process`. |
| **Reuse the same environment** | Once a process tensor is built, evolve new initial states with `evolve`, evaluate different protocols, or probe multi-time correlations without rebuilding the bath. |

## Installation

Install the latest tagged release from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/Gauthameshwar/ProcessTensors.jl", rev="v0.2.0")
```

For the latest development version:

```julia
using Pkg
Pkg.add(url="https://github.com/Gauthameshwar/ProcessTensors.jl")
```

### ACE: compress the environment, keep the memory

`ACE()` — Automated Compression of Environments — sequentially incorporates independent microscopic bath modes and compresses the temporal memory they leave behind.

That makes models such as a central spin surrounded by hundreds of bath spins, or a driven two-level system coupled to a thermal bosonic continuum, accessible without explicitly propagating the exponentially large joint environment (see the [Central-spin ACE example](https://Gauthameshwar.github.io/ProcessTensors.jl/stable/examples/central_spin_ace/) and [Thermal spin-boson ACE example](https://Gauthameshwar.github.io/ProcessTensors.jl/stable/examples/thermal_spinboson_ace/)
for complete walkthroughs).

## A process tensor in a few lines

Define the system and one bath mode:

```julia
using ITensors
using ProcessTensors

dt = 0.1
nsteps = 24

sys = siteinds("S=1/2", 1)
bath_site = siteinds("S=1/2", 1)
bathL = liouv_sites(bath_site)

Hsys = OpSum()
Hsys += 1.0, "Sx", 1
system = spin_system(sys, Hsys)

ρbath0 = to_liouville(
    to_dm(MPS(bath_site, ["Up"]));
    sites=bathL,
)

Hbath = OpSum()
Hbath += 1.0, "Sx", 1

Hint = OpSum()
Hint += 1.0, "Sz", 1, "Sz", 2

mode = spin_mode(bathL, Hbath, ρbath0; coupling=Hint)
environment = spin_bath([mode])

pt = build_process_tensor(
    system;
    environment,
    dt,
    nsteps,
)
```

Then reuse that process tensor however you like.

Follow the reduced non-Markovian trajectory:

```julia
ρ0 = to_dm(MPS(sys, ["Up"]))
trajectory = evolve(pt, ρ0)
```

Or ask an explicit experimental question:

```julia
obs = OpSum()
obs += 1.0, "Sz", 1

seq = default_schedule(pt)
add!(seq, 0, state_preparation(ρ0))
add!(seq, nsteps, observable_measurement(obs))

expectation = evaluate_process(pt, seq)
```

The process tensor is the reusable object in the middle: change the preparation,
measurement, control sequence, or multi-time probe without reconstructing the
environment.

## Pick your route through the docs

The documentation is written as a progression rather than an API dump.

- **New to the tensor-network conventions?** Start with [ITensor Basics](https://Gauthameshwar.github.io/ProcessTensors.jl/stable/tutorials/itensor_basics/) and [MPS and MPO Basics](https://Gauthameshwar.github.io/ProcessTensors.jl/stable/tutorials/mps_mpo_basics/).
- **Want open-system dynamics first?** Go to [Liouville-Space Basics](https://Gauthameshwar.github.io/ProcessTensors.jl/stable/tutorials/liouville_basics/) and [Dissipative Dynamics](https://Gauthameshwar.github.io/ProcessTensors.jl/stable/tutorials/dissipative_dynamics/).
- **Here for process tensors?** Start with the [Single-Mode Process Tensor tutorial](https://Gauthameshwar.github.io/ProcessTensors.jl/stable/tutorials/process_tensor_singlemode/), then move to the ACE examples.
- **Already know the theory?** Jump straight to the [Examples](https://Gauthameshwar.github.io/ProcessTensors.jl/stable/examples/tebd_time_evolution/) or the [API Reference](https://Gauthameshwar.github.io/ProcessTensors.jl/stable/api/).

## Contributing

`ProcessTensors.jl` is under active development. Bug reports, physics examples,
algorithm implementations, documentation improvements, and discussions about
future process-tensor methods are most welcome!

See [CONTRIBUTING.md](CONTRIBUTING.md) and [CHANGELOG.md](CHANGELOG.md) for
development and release information.

---

_No tensor indices were harmed during the development of this package. Several,
however, were accidentally contracted with the wrong ones before eventually
finding their soulmate._
