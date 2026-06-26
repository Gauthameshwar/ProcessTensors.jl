# Tensor Networks in Physics

Tensor networks are a way of representing very large quantum states and operators by breaking them into smaller tensors connected by shared indices. Instead of storing one exponentially large array, we store a structured network of local tensors. This is especially useful in one-dimensional many-body physics, where physically relevant states often contain much less information than the full Hilbert space allows.

This page gives only the minimum tensor-network background needed to understand the rest of the `ProcessTensors.jl` documentation. It is not meant to replace a full course or review article on tensor networks. For deeper study, use the references and online resources at the end of this page.

## Why tensor networks appear in many-body physics

A chain of $N$ spin-$1/2$ particles has a Hilbert space of dimension $2^N$. A general state is

```math
|\psi\rangle =
\sum_{s_1,\ldots,s_N}
c_{s_1\cdots s_N}
|s_1,\ldots,s_N\rangle,
```

where each $s_j \in \{\uparrow,\downarrow\}$. The coefficient tensor $c_{s_1\cdots s_N}$ has $2^N$ entries. For a local dimension $d$, this becomes $d^N$ entries.

Tensor networks ask a practical question:

> Can this large coefficient tensor be written as a contraction of smaller tensors?

For many physically relevant one-dimensional states, especially low-entanglement states, the answer is yes. The large tensor $c_{s_1\cdots s_N}$ is not stored directly. Instead, it is decomposed into local tensors connected by internal indices.

!!! note "The main idea"
    Tensor networks do not remove the exponential size of the full Hilbert space. They give an efficient representation for special but physically important parts of it.

## Matrix product states

The most common tensor network in this package is the **matrix product state**, or MPS. An MPS writes the coefficient tensor of a many-body state as

```math
c_{s_1\cdots s_N}
=
\sum_{\alpha_1,\ldots,\alpha_{N-1}}
A^{s_1}_{\alpha_1}
A^{s_2}_{\alpha_1\alpha_2}
A^{s_3}_{\alpha_2\alpha_3}
\cdots
A^{s_N}_{\alpha_{N-1}}.
```

Equivalently,

```math
|\psi\rangle =
\sum_{s_1,\ldots,s_N}
\sum_{\alpha_1,\ldots,\alpha_{N-1}}
A^{s_1}_{\alpha_1}
A^{s_2}_{\alpha_1\alpha_2}
\cdots
A^{s_N}_{\alpha_{N-1}}
|s_1,\ldots,s_N\rangle.
```

Each site has a physical index $s_j$, and neighbouring sites are connected by internal indices $\alpha_j$. These internal indices are often called **bond indices**, **link indices**, or **virtual indices**.

```text
physical legs:     s₁     s₂     s₃           sₙ
                   |      |      |            |
MPS:              [A] -- [A] -- [A] -- ... -- [A]
                    α₁     α₂             αₙ₋₁
```

The maximum size of the internal indices is called the **bond dimension**, often denoted by $\chi$.

A product state has bond dimension $\chi=1$. More entangled states require larger $\chi$. In an exact MPS representation, $\chi$ may still grow exponentially with system size. The useful regime is when the state can be accurately represented with moderate $\chi$.

Across a bipartition of the chain, a pure state can be written in Schmidt form as

```math
|\psi\rangle
=
\sum_{\alpha=1}^{r}
\lambda_\alpha
|\alpha_L\rangle
|\alpha_R\rangle.
```

The Schmidt rank $r$ tells us how many independent left-right components are needed across that cut. The MPS bond dimension across that cut must be large enough to store these components. This is why bond dimension is closely tied to entanglement.

!!! tip "Practical takeaway"
    In MPS simulations, the bond dimension is one of the main quantities to monitor. If the required bond dimension grows too quickly, the simulation becomes expensive or inaccurate.

## Matrix product operators

A **matrix product operator**, or MPO, is the operator analogue of an MPS. Instead of representing a many-body state, it represents a many-body operator such as a Hamiltonian, a time-evolution operator, a density matrix, or a Liouvillian superoperator.

A many-body operator has matrix elements

```math
O_{s_1'\cdots s_N', s_1\cdots s_N}.
```

For a chain with local dimension $d$, the full operator contains $d^{2N}$ matrix elements. This is already much larger than the $d^N$ coefficients needed for a pure state vector.

An MPO decomposes these operator coefficients into local tensors connected by bond indices:

```math
O
=
\sum_{s_1,\ldots,s_N,s_1',\ldots,s_N'}
\sum_{\beta_1,\ldots,\beta_{N-1}}
W^{s_1's_1}_{\beta_1}
W^{s_2's_2}_{\beta_1\beta_2}
\cdots
W^{s_N's_N}_{\beta_{N-1}}
|s_1'\cdots s_N'\rangle
\langle s_1\cdots s_N|.
```

```text
output legs:       s₁'    s₂'    s₃'          sₙ'
                   |      |      |            |
MPO:              [W] -- [W] -- [W] -- ... -- [W]
                   |      |      |            |
input legs:        s₁     s₂     s₃           sₙ
```

In tensor-network language, an MPO has two physical legs per site: one input leg and one output leg.

In `ProcessTensors.jl`, MPOs appear in several places:

* Hamiltonians are represented as MPOs.
* Density matrices can be represented as operator-like tensor networks in Hilbert space.
* Liouvillian superoperators are represented as MPOs in Liouville space.
* Process tensors are stored as tensor networks with physical input/output legs at each timestep, and memory links.

!!! note "MPS versus MPO"
    An MPS represents a vector-like object. An MPO represents a map-like object. Density matrices sit between these viewpoints: in Hilbert space they are operators, while in Liouville space they can be treated as vectorised states.

## Contractions

A tensor network becomes a number, state, operator, or reduced object by **contracting** shared indices. Contracting an index means summing over all values of that index.

For two tensors $A$ and $B$ sharing an index $\alpha$,

```math
C_{ij}
=
\sum_{\alpha}
A_{i\alpha}B_{\alpha j}.
```

This is just matrix multiplication written as an index contraction. Tensor networks generalise this idea to many indices and many tensors.

The inner product $\langle\phi|\psi\rangle$ is obtained by contracting every physical and bond index between the bra MPS and ket MPS.

```text
bra:              [B†] -- [B†] -- [B†] -- ... -- [B†]
                   |      |      |             |
ket:              [A]  -- [A]  -- [A]  -- ... -- [A]
```

In equations,

```math
\langle\phi|\psi\rangle
=
\sum_{s_1,\ldots,s_N}
\overline{\phi}_{s_1\cdots s_N}
\psi_{s_1\cdots s_N}.
```

Expectation values are contractions too:

```math
\langle O\rangle
=
\langle \psi|O|\psi\rangle.
```

In diagrammatic language, this means placing the MPO between the bra and ket MPS and contracting all matching legs.

```text
bra:              [A†] -- [A†] -- [A†]
                   |      |      |
operator:         [W] -- [W] -- [W]
                   |      |      |
ket:              [A]  -- [A]  -- [A]
```

This contraction viewpoint is important because `ProcessTensors.jl` uses the same idea for process tensors: a process tensor is evaluated by contracting it with a sequence of instruments.

## Truncation and approximation

Tensor-network simulations are powerful because they can compress information. This compression usually happens through singular-value decompositions.

Suppose a tensor is reshaped into a matrix $M$ across some chosen bipartition. Its singular-value decomposition is

```math
M = U S V^\dagger,
```

where $S$ contains non-negative singular values. If many singular values are very small, one can approximate $M$ by keeping only the largest ones:

```math
M
\approx
U_{\mathrm{kept}}
S_{\mathrm{kept}}
V_{\mathrm{kept}}^\dagger.
```

For an MPS, this operation is closely related to truncating the Schmidt decomposition

```math
|\psi\rangle =
\sum_{\alpha}
\lambda_\alpha
|\alpha_L\rangle |\alpha_R\rangle.
```

Keeping only the largest $\lambda_\alpha$ gives an approximate state with smaller bond dimension.

This is the basic compression step behind many tensor-network algorithms. In practice, simulations usually control truncation using parameters such as a maximum bond dimension and a singular-value cutoff.

!!! tip "Learn the SVD machinery of ITensors"
    The official [ITensors](https://docs.itensor.org/ITensors/stable/) documentation has examples of performing SVDs on matrices and higher-order tensors using named indices. This is a good place to learn how tensors are split, how singular values appear, and how contractions rebuild the original object.

## Tensor networks in `ProcessTensors.jl`

This package builds on the `ITensors.jl` and `ITensorMPS.jl` ecosystem. If you already know how to use `siteinds`, `MPS`, `MPO`, `OpSum`, `apply`, `expect`, or `tdvp`, then much of the syntax will feel familiar.

The package adds a layer of open-system structure on top of that familiar tensor-network language.

In particular, later pages will explain how `ProcessTensors.jl` uses tensor networks for:

* Hilbert-space dynamics,
* density matrices,
* Liouville-space vectorisation,
* Liouvillian MPOs,
* process tensor construction,
* instruments and interventions,
* reduced dynamics,
* multi-time observables.

!!! note "Why borrow tensor networks for open quantum simulation?"
    Open quantum systems are usually described by density matrices rather than pure wavefunctions. For a chain with local dimension $d$, a pure state has $d^N$ amplitudes, while a density matrix has $d^{2N}$ coefficients. This squared scaling makes exact density-matrix simulation much harder than closed-system wavefunction simulation.

    Tensor networks provide a compression strategy. Instead of storing the full density matrix or Liouvillian, one can represent them as matrix product density operators, Liouville-space MPS/MPOs, or related tensor-network objects. This does not make every open-system problem easy, but it gives a controlled language for approximating mixed states, dissipative evolution, and memory effects using bond dimensions and truncation cutoffs.

    In `ProcessTensors.jl`, this idea appears in two ways: density matrices can be lifted into Liouville space and evolved with Liouvillian MPOs, and process tensors can store system-bath memory in tensor-network bonds.

    For more background, see the [Mixed states, MPDOs, and open-system tensor networks](#mixed-states-mpdos-and-open-system-tensor-networks), as well as the [Liouville-space theory page](liouville_space.md) in this documentation.

The goal is not to replace `ITensorMPS.jl`, but to extend its style of computation toward open quantum systems and non-Markovian processes.

## Further reading

This page only provides the vocabulary needed for the rest of the documentation. For a more detailed read, check out the resources below.

### Package documentation and visual guides

1. [ITensors.jl documentation](https://docs.itensor.org/ITensors/stable/)
   
   Best for named tensor indices, contractions, tensor SVDs, and the basic `ITensor` object.

2. [ITensorMPS.jl documentation](https://docs.itensor.org/ITensorMPS/stable/)
   
   Best for practical Julia usage of `MPS`, `MPO`, `OpSum`, DMRG, and MPS time evolution.

3. [TensorNetwork.org](https://tensornetwork.org/)
   
   A broad community resource with introductory and review-style material on tensor networks, algorithms, and software.

4. [Tensors.net](https://www.tensors.net/)
   
   Useful for visual tensor-network tutorials, especially if you want to understand diagrams, contractions, decompositions, and algorithmic building blocks.

### Introductory papers and reviews

1. [Roman Orús, “A Practical Introduction to Tensor Networks: Matrix Product States and Projected Entangled Pair States”](https://arxiv.org/abs/1306.2164)
   
    A beginner-friendly conceptual introduction to tensor networks, MPS, and PEPS.

2. [Jacob C. Bridgeman and Christopher T. Chubb, “Hand-waving and Interpretive Dance: An Introductory Course on Tensor Networks”](https://arxiv.org/abs/1603.03039)
   
   A readable introduction emphasizing graphical tensor-network reasoning.

3. [Jacob Biamonte and Ville Bergholm, “Tensor Networks in a Nutshell”](https://arxiv.org/abs/1708.00006)
   
   A compact overview of tensor-network ideas and notation.

4. [Ulrich Schollwöck, “The density-matrix renormalization group in the age of matrix product states”](https://arxiv.org/abs/1008.3477)
   
   A deeper review of MPS, canonical forms, DMRG, and one-dimensional quantum systems.

### Mixed states, MPDOs, and open-system tensor networks

1. [F. Verstraete, J. J. García-Ripoll, and J. I. Cirac, “Matrix Product Density Operators: Simulation of finite-T and dissipative systems.”](https://arxiv.org/abs/cond-mat/0406426)  
   
   A foundational reference introducing matrix product density operators as tensor-network representations of mixed states. 

2. [D. Jaschke, S. Montangero, and L. D. Carr, “One-dimensional many-body entangled open quantum systems with tensor network methods.”](https://arxiv.org/abs/1804.09796)  
   
   A broad and accessible entry point for open-system tensor-network simulations.

3. [J. G. Jarkovsky, A. Molnar, N. Schuch, and J. I. Cirac, “Efficient description of many-body systems with Matrix Product Density Operators.”](https://arxiv.org/abs/2003.12418)  
   
   A more theoretical reference about MPDO representation and when mixed quantum states admit efficient MPDO descriptions.