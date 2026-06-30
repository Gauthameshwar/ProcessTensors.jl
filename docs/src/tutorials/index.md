# Tutorials

The tutorials are guided, executable walkthroughs. They build from ITensor syntax to Hilbert-space MPS/MPO objects, then to Liouville-space dynamics, dissipative evolution, and finally process tensors.

They are generated from Literate.jl sources in `docs/literate/tutorials/`, so the code shown on each page is meant to be runnable.

```text
ITensor Basics
    → MPS and MPO Basics
    → Liouville-Space Basics
    → Unitary Dynamics
    → Dissipative Dynamics
    → Single-Mode Process Tensor
```

| Tutorial | What you learn | Why it matters |
| --- | --- | --- |
| [ITensor Basics](itensor_basics.md) | Index identity, contractions, priming, delta tensors, and SVD. | These are the mechanics behind every MPS/MPO contraction. |
| [MPS and MPO Basics](mps_mpo_basics.md) | `siteinds`, `MPS`, `MPO`, `OpSum`, expectations, density matrices, and reduced states. | This is the Hilbert-space tensor-network foundation used by the rest of the package. |
| [Liouville-Space Basics](liouville_basics.md) | `liouv_sites`, `to_liouville`, trace/expectation overlaps, and Liouville superoperators. | This fixes the density-matrix and vectorisation conventions used for open-system dynamics. |
| [Unitary Dynamics](unitary_dynamics.md) | Closed-system TEBD/TDVP and the Hilbert-versus-Liouville comparison. | It connects familiar pure-state time evolution to density-matrix evolution. |
| **[Dissipative Dynamics](dissipative_dynamics.md)** | Lindblad generators, `MPO_Liouville`, and TEBD/TDVP on density matrices. | This is one of the two core workflows of `ProcessTensors.jl`. |
| **[Single-Mode Process Tensor](process_tensor_singlemode.md)** | Bath modes, process-tensor construction, reduced evolution, instruments, and correlations. | This is the main process-tensor workflow and the best end-to-end open-system example. |

!!! note "Tutorials are cumulative"
    Later tutorials assume the syntax and conventions introduced earlier, especially ITensor index identity, Liouville-site reuse, and density-matrix vectorisation.

!!! tip "Already know ITensors?"
    Start at [Liouville-Space Basics](liouville_basics.md), then move to [Dissipative Dynamics](dissipative_dynamics.md) and [Single-Mode Process Tensor](process_tensor_singlemode.md).

For notation and conceptual background, use the [Theory](../theory/index.md) pages. For compact model templates, use [Examples](../examples/index.md).
