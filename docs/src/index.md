# ProcessTensors.jl

`ProcessTensors.jl` is an ITensor-native framework for studying quantum systems
with environmental memory. Construct a process tensor from a microscopic
environment, then reuse it with different preparations, controls, measurements,
and multi-time probes. 

These tutorials connect the physical picture, tensor-network representation,
and runnable Julia code, helping you learn the process-tensor framework while
using it to design numerical experiments.

## Build once, explore many experiments

When a system interacts with an environment, its present reduced state may not
contain all the information needed to predict its future. The environment can
retain information about earlier interactions and interventions. A process tensor
describes how the system responds to interventions at different times, including
the influence of environmental memory. ProcessTensors.jl represents this
multi-time response using matrix product operators, allowing the same process
tensor to be reused across different experiments. 

```@raw html
<figure class="feature-clip">
  <img class="theme-figure-light clip-motion" loading="lazy" src="assets/animations/ace-light.gif" alt="Time runs right to left. Two environment modes over three time steps are absorbed column by column and compressed into a three-core process tensor with downward system legs.">
  <img class="theme-figure-dark clip-motion" loading="lazy" src="assets/animations/ace-dark.gif" alt="Time runs right to left. Two environment modes over three time steps are absorbed column by column and compressed into a three-core process tensor with downward system legs.">
  <img class="theme-figure-light clip-poster" loading="lazy" src="assets/animations/ace-light-poster.png" alt="A three-core process tensor with downward system legs, produced by ACE compression of two environment modes.">
  <img class="theme-figure-dark clip-poster" loading="lazy" src="assets/animations/ace-dark-poster.png" alt="A three-core process tensor with downward system legs, produced by ACE compression of two environment modes.">
  <figcaption><strong>Construct a process tensor:</strong> Combine the influence of independent environmental modes and compress the resulting temporal bonds using ACE.</figcaption>
</figure>

<figure class="feature-clip">
  <img class="theme-figure-light clip-motion" loading="lazy" src="assets/animations/instruments-light.gif" alt="Time runs right to left. A preparation, a custom map, and an open output snap into the slots of a three-step instrument tape; the unspecified slot is filled by an identity.">
  <img class="theme-figure-dark clip-motion" loading="lazy" src="assets/animations/instruments-dark.gif" alt="Time runs right to left. A preparation, a custom map, and an open output snap into the slots of a three-step instrument tape; the unspecified slot is filled by an identity.">
  <img class="theme-figure-light clip-poster" loading="lazy" src="assets/animations/instruments-light-poster.png" alt="A filled three-step instrument tape: preparation, identity, custom map, and open output.">
  <img class="theme-figure-dark clip-poster" loading="lazy" src="assets/animations/instruments-dark-poster.png" alt="A filled three-step instrument tape: preparation, identity, custom map, and open output.">
  <figcaption><strong>Customize your instruments:</strong> Define the interventions you wish to perform on the system at chosen times; unspecified slots default to identity.</figcaption>
</figure>

<figure class="feature-clip">
  <img class="theme-figure-light clip-motion" loading="lazy" src="assets/animations/contraction-light.gif" alt="Time runs right to left. Two identical process tensors are contracted with instruments; the left ends as an open reduced state (triangle) and the right as a closed scalar (circle).">
  <img class="theme-figure-dark clip-motion" loading="lazy" src="assets/animations/contraction-dark.gif" alt="Time runs right to left. Two identical process tensors are contracted with instruments; the left ends as an open reduced state (triangle) and the right as a closed scalar (circle).">
  <img class="theme-figure-light clip-poster" loading="lazy" src="assets/animations/contraction-light-poster.png" alt="Contraction results: an open reduced state (triangle) on the left and a closed scalar (circle) on the right.">
  <img class="theme-figure-dark clip-poster" loading="lazy" src="assets/animations/contraction-dark-poster.png" alt="Contraction results: an open reduced state (triangle) on the left and a closed scalar (circle) on the right.">
  <figcaption><strong>Evaluate and reuse:</strong> Contract the same process tensor with different instruments to obtain reduced states and scalar values.</figcaption>
</figure>
```

## Get started

Follow [Installation](installation.md) to set up the package, then choose your learning route depending on what suits you the best

| Your starting point | Suggested route |
| --- | --- |
| **Ready to use process tensors** | [Installation](installation.md), then [Construct your first process tensor](tutorials/process_tensor_singlemode.md) and [Explore a process with instruments](tutorials/process_tensor_instruments.md). Consult the [API reference](api.md) as needed. |
| **Learning the physical framework** | Start with [Process Tensors](theory/process_tensors.md), then [Construct your first process tensor](tutorials/process_tensor_singlemode.md). |
| **New to ITensor or Liouville representations** | Use [ITensor Basics](tutorials/itensor_basics.md), [MPS and MPO Basics](tutorials/mps_mpo_basics.md), and [Liouville-Space Basics](tutorials/liouville_basics.md) for the supporting conventions. |

The foundations explain named ITensor indices, density matrices, and vectorisation.
You can consult them whenever these concepts arise in a process-tensor
calculation. 

For theory and notation, see [Process Tensors](theory/process_tensors.md),
[Quantum States and Liouville Space](theory/liouville_space.md), and
[Tensor Networks in Physics](theory/tensor_networks.md). For progress reporting,
threading, and execution settings, see [Advanced Usage](advanced_usage.md).

## What can you explore?

| What can you explore? | Suggested walkthrough |
| --- | --- |
| Build a process from several environmental modes | [Spin-bath process tensor](examples/spin_bath_process_tensor.md) |
| Compress spin and thermal bosonic environments with ACE | [Central-spin dynamics using ACE](examples/central_spin_ace.md); [Thermal spin-boson dynamics using ACE](examples/thermal_spinboson_ace.md) |
| Design preparations, controls, and measurements | [Explore a process with instruments](tutorials/process_tensor_instruments.md) |
| Probe bath memory through repeated measurements | [Ramsey readouts as a probe of bath memory](examples/ramsey_povm.md) |
| Calculate multi-time correlations | [Multi-time correlations](examples/multitime_correlations.md) |
| Carry an ancillary system between interventions | [Testers and noisy quantum circuit](examples/noisy_quantum_circuit_tester.md) |

Use `Dense()` for small environments or `ACE()` to incorporate initially
independent bath modes and compress their temporal influence. Gaussian PT-TEMPO
and chain-mapped constructions are planned extensions.

## Additional tools for open quantum dynamics

The package also provides Hilbert- and Liouville-space MPS/MPO objects,
density-matrix conversions, Lindblad generators, and TEBD/TDVP evolution.
These tools support model preparation and reference calculations and can also
be used independently for time-local dynamics.

See [Unitary Dynamics](tutorials/unitary_dynamics.md) for closed-system
evolution and [Dissipative Dynamics](tutorials/dissipative_dynamics.md) for
Markovian open-system evolution. Larger models include a
[dissipative spin chain](examples/dissipative_spin.md) and
[driven-dissipative Bose–Hubbard dynamics](examples/driven_dissipative_bose_hubbard.md).

## Citing and contributing

If you use `ProcessTensors.jl` in research, please cite the
[package repository](https://github.com/Gauthameshwar/ProcessTensors.jl) and the
relevant methods and theory references linked in the documentation.

Bug reports, questions, new examples, and algorithm contributions are welcome.
Visit the [issue tracker](https://github.com/Gauthameshwar/ProcessTensors.jl/issues)
for questions and feedback, or read the
[contribution guidelines](https://github.com/Gauthameshwar/ProcessTensors.jl/blob/main/CONTRIBUTING.md)
to get involved.
