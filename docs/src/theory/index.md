# Theory

These pages explain the concepts and conventions used throughout `ProcessTensors.jl`: tensor networks, density matrices, Liouville-space vectorisation, and process tensors.

They provide the minimum conceptual and notational background needed to understand the tutorials. They are meant as signposts, not as a replacement for a textbook on tensor networks or open quantum systems.

```text
Tensor networks
    → density matrices
    → Liouville space
    → process tensors
```

| Page | Read this if | Main takeaway |
| --- | --- | --- |
| [Tensor Networks in Physics](tensor_networks.md) | You want the MPS/MPO language used in the package. | Tensor networks compress large states, operators, density matrices, and process objects using local tensors and bond indices. |
| [Quantum States and Liouville Space](liouville_space.md) | You want the Hilbert-space, density-matrix, and vectorisation conventions. | Density matrices can be represented as Liouville-space MPS objects, while channels and Liouvillians become MPO-like maps. |
| [Process Tensors](process_tensors.md) | You want the multi-time open-system picture. | Process tensors store environment memory and are evaluated by inserting instruments into time slots. |

!!! tip "Use theory as a convention dictionary"
    These pages are not prerequisites for every tutorial. If you already know the physics, use them mainly to check notation, vectorisation conventions, input/output legs, and the meaning of instruments.

For runnable workflows, go to [Tutorials](../tutorials/index.md). For model-oriented scripts, go to [Examples](../examples/index.md).
