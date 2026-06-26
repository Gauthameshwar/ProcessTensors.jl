# Process Tensors

A process tensor is the central object used to describe a quantum system that is probed, controlled, measured, or otherwise intervened on at multiple times while it remains coupled to an environment.

The goal of this page is to explain what a process tensor is, why it naturally describes non-Markovian open quantum dynamics, and how the same object appears in `ProcessTensors.jl` as a tensor network with time steps, input/output legs, memory links, and instruments.

For the Hilbert-space, density-matrix, and Liouville-space conventions used in this package, see [Quantum States and Liouville Space](liouville_space.md).

## From channels to multi-time processes

A quantum state is represented by a density matrix $\rho$. A quantum channel is a physical transformation that maps input states to output states:

```math
\rho'
=
\Phi(\rho).
```

For deterministic dynamics, $\Phi$ is usually required to be Completely Positive and Trace Preserving (CPTP). The Local Master Equation, and the closed unitary dynamics of $\rho$ given by Von-Neuman equations are examples of channels.

In this sense, an ordinary channel answers a one-time question: Given the state now, what is the state later?

We can extend this and ask a more operational question: Given the operations performed on the system at several earlier times, what state, or outcome do we obtain later?

To express this idea cleanly, it is useful to introduce the language of **superchannels**. A channel maps states to states. A superchannel is a higher-order transformation that takes quantum channels as inputs and returns a state output. In this sense, a process tensor is a multi-time superchannel: it takes the operations inserted at different times as its inputs.

Schematically,

```math
\mathcal{T}_{n:0}
[
\mathcal{A}_{n-1},
\ldots,
\mathcal{A}_0
]
=
\rho_n.
```

Here $\mathcal{A}_k$ is the operation inserted at time $t_k$, and $\rho_n$ is the resulting output state at the final time $t_n$. If the final output is also measured or traced out, the same process tensor gives a scalar probability or expectation value instead.

```text

Multi-time process tensor:

      A₀        A₁        A₂
      │         │         │
      ▼         ▼         ▼
    [ process tensor with memory ] ──> ρ₃
```

!!! note "Operations, channels, and instruments"
    - A **quantum operation** is a completely positive map that may decrease the trace of the density matrix. It can represent a probabilistic event, such as one outcome of a measurement.

    - A **quantum channel** is a deterministic operation. It is completely positive and trace preserving.

    - An **instrument** is a collection of operations labelled by outcomes, $\mathcal{J}=\{\mathcal{A}^{(x)}\}_x$. Each $\mathcal{A}^{(x)}$ is a completely positive map associated with outcome $x$, and the sum over all outcomes is trace preserving.

    In process-tensor language, an instrument does not mean only a measurement. It can represent preparation, evolution, measurement, trace-out, or a custom operation placed into a time slot of the process tensor.

A process tensor over times $t_0,\ldots,t_n$ can therefore be read as a higher-order object that accepts a sequence of operations:

```math
\mathcal{T}_{n:0}
:
(
\mathcal{A}_{n-1},
\ldots,
\mathcal{A}_0
)
\mapsto
\rho_n.
```

Equivalently,

```math
\rho_n
=
\mathcal{T}_{n:0}
[
\mathcal{A}_{n-1},
\ldots,
\mathcal{A}_0
].
```

If the operations are selected from instruments with outcomes $x_0,\ldots,x_{n-1}$, and the final output is closed, the process tensor gives a joint probability:

```math
p(x_{n-1},\ldots,x_0)
=
\mathcal{T}_{n:0}
[
\mathcal{A}_{n-1}^{(x_{n-1})},
\ldots,
\mathcal{A}_0^{(x_0)}
].
```

In a Choi-state representation, the process tensor is often denoted by $\Upsilon_{n:0}$. The corresponding generalised Born rule has the schematic form

```math
p(\mathbf{x})
=
\operatorname{Tr}
\left[
\Upsilon_{n:0}
\left(
\mathsf{A}_{n-1}^{(x_{n-1})}
\otimes
\cdots
\otimes
\mathsf{A}_0^{(x_0)}
\right)
\right],
```

up to transpose and index-ordering conventions fixed by the chosen Choi or Liouville representation. Here $\mathsf{A}_k^{(x_k)}$ denotes the Choi or Liouville representation of the operation inserted at time $t_k$.

!!! warning "Conventions matter"
    Process tensors can be written using Choi states, Liouville vectors, quantum combs, or diagrammatic tensor notation. These representations describe the same operational object, but transpose and index-ordering conventions differ. In this package, the Liouville-space convention is described in [Quantum States and Liouville Space](liouville_space.md).

## Non-Markovian dynamics and process tensors

If an open quantum system is effectively memoryless, one often describes its reduced dynamics by a chain of state-to-state maps:

```math
\rho_{k+1}
=
\Phi_{k+1:k}
\left[
\rho_k
\right].
```

This is the Markovian intuition: the future depends only on the present reduced state.

But for a system coupled to a bath, the system and bath can become correlated. If the bath retains information about previous system states or previous interventions, then the transformation from $t_k$ to $t_{k+1}$ is not determined only by $\rho_k$.

In such a case, the later state may depend on the full history of operations:

```math
\rho_n
=
\mathcal{T}_{n:0}
[
\mathcal{A}_{n-1},
\ldots,
\mathcal{A}_0
].
```

This is the operational meaning of memory: later statistics can depend on what was done to the system at earlier times.

```text
Markovian picture:

ρ₀ ── Φ₁ ──> ρ₁ ── Φ₂ ──> ρ₂ ── Φ₃ ──> ρ₃


Non-Markovian process picture:

A₀        A₁        A₂
│         │         │
▼         ▼         ▼
[slot 0]─m₁─[slot 1]─m₂─[slot 2]─m₃─> output

memory links carry history
```

!!! note "Memory as operational dependence"
    Non-Markovianity is not only the statement that “the bath has memory.” In the process-tensor framework, memory means that future states or probabilities can depend on the sequence of previous interventions.

The most common physical origin of a process tensor is a system $S$ interacting with a bath $B$. Suppose the joint system-bath state at the initial time is $\rho_{SB}(t_0)$. Between interventions, the joint state evolves under system-bath propagators. In superoperator notation,

```math
\mathcal{U}_{k+1:k}(\cdot)
=
U_{k+1:k}
(\cdot)
U_{k+1:k}^{\dagger}.
```

At each time $t_k$, the user may apply an operation $\mathcal{A}_k$ to the system. The bath is not directly controlled, so the operation acts as $\mathcal{A}_k\otimes\mathcal{I}_B$ on the joint state.

The final reduced system state is

```math
\rho_S(t_n)
=
\operatorname{Tr}_B
\left[
\mathcal{U}_{n:n-1}
\left(
\mathcal{A}_{n-1}\otimes\mathcal{I}_B
\right)
\cdots
\mathcal{U}_{1:0}
\left(
\mathcal{A}_{0}\otimes\mathcal{I}_B
\right)
\rho_{SB}(t_0)
\right].
```

This equation is the key construction. The process tensor is what remains when all the system-bath evolution and bath trace are packaged into a reusable object with open slots for the system operations $\mathcal{A}_k$.

```text
System-bath origin of a process tensor:

ρ_SB(t₀)
   │
   ▼
[A₀ on S] ── U₁:₀ ── [A₁ on S] ── U₂:₁ ── [A₂ on S] ── U₃:₂
                                                               │
                                                               ▼
                                                         Tr_B at end

The bath is not directly controlled, but it carries memory between time steps.
```

A Markovian process is a special case of the process-tensor framework. In this case, the multi-time process factorises into independent step-to-step channels:

```math
\mathcal{T}_{n:0}^{\mathrm{Markov}}
\sim
\Phi_{n:n-1}
\otimes
\Phi_{n-1:n-2}
\otimes
\cdots
\otimes
\Phi_{1:0}.
```

There is no non-trivial memory link carrying information between interventions. In a non-Markovian process, this factorisation fails. The memory links are non-trivial, and the effect of an operation at one time can influence later reduced dynamics.

```text
Markovian temporal structure:

[Φ₁]   [Φ₂]   [Φ₃]

no memory links between interventions


Non-Markovian temporal structure:

[PT₀]──m₁──[PT₁]──m₂──[PT₂]──m₃──[PT₃]

memory links carry temporal correlations
```

!!! note "Memory and bond dimension"
    In a tensor-network representation, memory is reflected in temporal correlations and memory bond dimensions. A compact memory bond does not mean there is no memory; it means the memory is efficiently compressible.

## Anatomy of a process tensor

In tensor-network form, a process tensor has an input and output leg at each time slot.

The input leg represents the system state entering that time slot. The output leg represents the system state leaving that time slot. The operation inserted by the user connects the input-output structure associated with that time.

A memory link connects neighbouring process-tensor cores. It stores compressed information about the bath influence and the history of previous interactions.

```text
One process-tensor core:

              output leg
                  │
left memory ── [ PTₖ ] ── right memory
                  │
               input leg
```

A multi-time process tensor becomes a chain in time:

```text
Process tensor as a temporal tensor network:

out₀       out₁       out₂       out₃
 │          │          │          │
[PT₀]--m₁--[PT₁]--m₂--[PT₂]--m₃--[PT₃]
 │          │          │          │
in₀        in₁        in₂        in₃
```

The memory bond dimensions determine how much temporal correlation is stored. A small memory bond corresponds to a compact process. A large memory bond means the bath influence or intervention history is harder to compress.

!!! note "Temporal tensor network"
    An MPS stores spatial correlations along a chain. A process tensor stores temporal correlations along a sequence of time steps. This is why process tensors can often be represented as matrix-product objects in time.

### Instruments in process-tensor slots

An instrument is inserted into the open legs of a process tensor. It represents the operation performed on the system at a given time.

Common examples include:

| Role                   | Meaning                                                     |
| ---------------------- | ----------------------------------------------------------- |
| State preparation      | Prepare the system in a chosen initial state                |
| Identity operation     | Let the system pass through the slot without intervention   |
| System propagation     | Insert a chosen system map or unitary                       |
| Observable measurement | Contract with an observable to compute an expectation value |
| Trace-out              | Close a leg with the identity effect                        |
| Open output            | Leave a leg uncontracted to obtain a reduced object         |
| Left/right insertion   | Insert operator actions such as $A\rho$ or $\rho A$         |
| Custom operation       | Insert a user-defined two-leg map                           |

```text
Instrument contraction:

        A₀          A₁          A₂
        │           │           │
out₀    │   out₁    │   out₂    │
 │      │    │      │    │      │
[PT₀]--m₁--[PT₁]--m₂--[PT₂]--m₃--[PT₃]
 │           │           │
in₀         in₁         in₂
```

The process tensor itself stores the environment-mediated multi-time structure. The instruments specify what the user does to the system.

!!! warning "Instrument is broader than measurement"
    A measurement is one kind of instrument. But in a process tensor calculation, an instrument can also mean preparation, propagation, trace-out, observable insertion, or leaving a leg open.

### Evaluating a process tensor

Evaluating a process tensor means contracting it with an instrument sequence.

If all relevant legs are closed, the result is a scalar. This scalar may be a probability, expectation value, or correlation value depending on the inserted instruments.

```text
All legs closed:

[A₀]      [A₁]      [A₂]      [A₃]
 │         │         │         │
[PT₀]--m--[PT₁]--m--[PT₂]--m--[PT₃]
 │         │         │         │
closed    closed    closed    closed

Result: scalar
```

If one output is left open, the result is a reduced system object at that time.

```text
One output left open:

[A₀]      [A₁]       open
 │         │          │
[PT₀]--m--[PT₁]--m--[PT₂]
 │         │          │
closed    closed     input/trace structure

Result: reduced state or Liouville-space object
```

Reduced time evolution is a special process-tensor contraction pattern. If the first instrument prepares an initial state, and later slots are filled with identity, trace-out, or propagation instruments, the process tensor returns the reduced system trajectory.

```text
Reduced evolution picture:

ρ₀ ── [PT₀] ──> ρ₁ ── [PT₁] ──> ρ₂ ── [PT₂] ──> ρ₃
```

In package language, this is the difference between the general and convenience workflows:

```julia
# General process contraction
evaluate_process(pt, seq)

# Common reduced-evolution workflow
evolve(pt, ρ0)
```

The second is not a different physical theory. It is a frequently used contraction of the same process tensor.

### Multi-time observables and correlations

Process tensors are useful for multi-time quantities because they keep intervention slots open. For example, a two-time quantity may be computed by inserting one operation at $t_1$ and another at $t_2$.

```text
Two-time operator insertion:

              B insertion              A insertion / trace
                   │                         │
[PT₀]──m──[PT₁]──m──[PT₂]──m──[PT₃]──m──[PT₄]
```

The precise instrument depends on what correlation is desired. A projective measurement, an observable insertion, and a left/right operator action are not the same operation.

!!! warning "Sequential measurements versus operator correlations"
    A sequential measurement correlation is built from actual measured joint probabilities and includes measurement backaction. An operator correlation such as $\langle A(t_2)B(t_1)\rangle$ is an operator-insertion object. These two quantities agree only under specific assumptions.

    This distinction matters because process tensors are operational. They respond to the actual instruments inserted into the time slots.

### Physical properties

A physical process tensor is not an arbitrary tensor. It must satisfy structural conditions inherited from quantum mechanics and causality.

* **Complete positivity.** If completely positive operations are inserted into the process tensor, the resulting state or probability must be physical. In Choi-like representations, this is reflected by positivity of the process tensor object.

* **Normalisation.** If a complete deterministic set of operations is inserted at every time, probabilities must normalise correctly. Equivalently, inserting trace-preserving operations should not create or destroy total probability.

* **Causality.** Future interventions cannot change earlier observed statistics. Operationally, if all future slots are closed with trace-preserving operations, then the probabilities for earlier outcomes must remain unchanged.

```text
Causality intuition:

past instruments       current statistics       future slots closed
      A₀, A₁                   ?                   Tr, Tr, Tr
        │                      │                    │   │   │
     [PT₀]──m──[PT₁]──m──[PT₂]──m──[PT₃]──m──[PT₄]──m──[PT₅]

Future trace-preserving closures cannot signal backwards in time.
```

!!! note "Causality is built into physical processes"
    A valid process tensor respects the arrow of time. Later choices may affect later results, but they cannot alter statistics that have already been fixed at earlier times.

### How this appears in `ProcessTensors.jl`

`ProcessTensors.jl` constructs process tensors from microscopic ingredients such as system sites, bath modes, system-bath couplings, time steps, and propagation rules.

The package workflow is conceptually:

```text
define system + bath + coupling
          │
          ▼
build_process_tensor(...)
          │
          ▼
ProcessTensor
          │
   ┌──────┴─────────┐
   ▼                ▼
evolve(...)    evaluate_process(...)
```

The important package objects are:

| Object or function     | Conceptual role                                                    |
| ---------------------- | ------------------------------------------------------------------ |
| `ProcessTensor`        | Tensor-network representation of a multi-time open quantum process |
| `build_process_tensor` | Constructs the process tensor from system-bath dynamics            |
| `InstrumentSeq`        | Stores the sequence of instruments inserted at time steps          |
| `evaluate_process`     | Contracts the process tensor with the instrument sequence          |
| `evolve`               | Convenience workflow for reduced system evolution                  |
| `OpenOutput`           | Leaves an output leg open to return a reduced object               |
| `TraceOut`             | Closes a leg by tracing out the corresponding degree of freedom    |

The theory is the same regardless of the implementation details: the process tensor stores the reusable environment influence, and instruments specify what is done to the system.

!!! note "Package perspective"
    In `ProcessTensors.jl`, the process tensor is represented as a tensor network in time. Its open legs are the places where instruments are inserted, and its memory links store compressed temporal correlations.

!!! warning "Common process-tensor mistakes"
    A process tensor is not just a time-evolution operator. It is a multi-time object that accepts operations at several times.

    A process tensor is not only a final state. Depending on which legs are contracted or left open, it can return probabilities, expectation values, reduced states, or Liouville-space objects.

    An instrument is not necessarily a measurement. It can represent preparation, propagation, trace-out, identity, observable insertion, or a custom operation.

    Leaving a leg open is different from tracing it out. An open leg returns an object; a traced-out leg closes part of the tensor network.

    Sequential measurement correlations are not automatically the same as theoretical operator correlations. Measurement backaction matters.

    A small memory bond does not always mean the process is Markovian. It may mean the non-Markovian memory is efficiently compressible.

## Further reading

This page is a compact guide to the process-tensor language used in the package. For deeper study, the following references are useful.

### Process tensors and quantum stochastic processes

1. [F. A. Pollock, C. Rodríguez-Rosario, T. Frauenheim, M. Paternostro, and K. Modi, “Non-Markovian quantum processes: complete framework and efficient characterisation.”](https://arxiv.org/abs/1512.00589)

   Foundational process-tensor reference. Useful for understanding how multi-time quantum processes with memory can be characterised operationally and represented as many-body quantum states.

2. [F. A. Pollock, C. Rodríguez-Rosario, T. Frauenheim, M. Paternostro, and K. Modi, “Operational Markov condition for quantum processes.”](https://arxiv.org/abs/1801.09811)

   Important reference for the operational definition of quantum Markovianity and the causal structure of process tensors.

3. [S. Milz and K. Modi, “Quantum stochastic processes and quantum non-Markovian phenomena.”](https://arxiv.org/abs/2012.01894)

   A pedagogical tutorial connecting classical stochastic processes, quantum combs, process tensors, and non-Markovian quantum phenomena.

### Higher-order quantum operations and quantum combs

1. [P. Taranto, S. Milz, M. Murao, M. T. Quintino, and K. Modi, “Higher-Order Quantum Operations.”](https://arxiv.org/abs/2503.09693)

   A broad review article on higher-order quantum operations, including superchannels, combs, process tensors, and their physical applications.

2. [G. Chiribella, G. M. D'Ariano, and P. Perinotti, “Theoretical framework for quantum networks.”](https://arxiv.org/abs/0904.4483)

   A foundational reference for quantum combs, link products, and higher-order transformations of quantum operations.

3. [G. Chiribella, G. M. D'Ariano, and P. Perinotti, “Quantum Circuit Architecture.”](https://arxiv.org/abs/0712.1325)

   Useful background for understanding how networks of quantum operations can be treated as higher-order quantum objects.

### Tensor-network process-tensor methods

1. [A. Strathearn, P. Kirton, D. Kilda, J. Keeling, and B. W. Lovett, “Efficient non-Markovian quantum dynamics using time-evolving matrix product operators.”](https://arxiv.org/abs/1711.09641)

   Introduces the TEMPO approach, which represents environment influence using matrix product operators in time.

2. [M. Cygorek, M. Cosacchi, A. Vagov, V. M. Axt, B. W. Lovett, J. Keeling, and E. M. Gauger, “Numerically exact open quantum systems simulations for arbitrary environments using automated compression of environments.”](https://arxiv.org/abs/2101.01653)

   Introduces automated compression of environments, a process-tensor-based method for constructing and compressing environmental influence.

3. [G. E. Fux, D. Kilda, B. W. Lovett, and J. Keeling, “Tensor network simulation of chains of non-Markovian open quantum systems.”](https://arxiv.org/abs/2201.05529)

   Useful follow-up for readers interested in combining process tensors with tensor-network simulations of interacting chains.
