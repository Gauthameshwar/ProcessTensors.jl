# SciPost Figure 1: correctness and discretisation

This folder generates the four-panel accuracy figure comparing direct
system-plus-environment exact diagonalisation (ED), the `Dense()` process
tensor builder, and ACE.

The physical model is the four-mode Sz⊗Sz spin bath from
[`scripts/pt_tfim_multimode.jl`](../../scripts/pt_tfim_multimode.jl). Data
generation and plotting are separate: the run scripts write CSV files under
`results/`; the single plotter never rebuilds a process tensor.

## Four-mode Sz⊗Sz model

We set $\hbar=1$. ITensor spin operators are $S\_\alpha=\sigma\_\alpha/2$;
the plotted observable is $\langle\sigma\_y(t)\rangle$.

$$
H=S\_x^{(S)}+\sum\_{m=1}^{4}
\left[\omega\_m S\_x^{(B\_m)}+g\_m S\_z^{(B\_m)}S\_z^{(S)}\right],
$$

with $\omega\_m=0.5+0.1m$ and $g\_m=0.2+0.3m$. Every spin starts in `Up`.
The joint Hilbert-space dimension is $2^5=32$.

The default final time is $T=6$.

Terminology:

- **Direct ED** diagonalises the complete 32-dimensional Hilbert-space
  Hamiltonian. It is independent of $\Delta t$.
- **Exact PT** is `method=Dense(), alg=Exact()`.
- **ACE** is `ACE(cutoff=1e-12, compression=:canonzip)`.

Time alignment matches the example script, not `evolve`'s labelled times:
$t=0$ is the unevolved product state, and process-tensor snapshot $k$ is
compared with ED at $t=k\Delta t$. The process tensor therefore has
$n\_{\mathrm{steps}}=T/\Delta t$ slabs.

## Generate data and figure

From the repository root:

```bash
OPENBLAS_NUM_THREADS=1 julia -t auto --project=benchmark benchmark/scipost_fig1/run_fig1_left.jl
OPENBLAS_NUM_THREADS=1 julia -t auto --project=benchmark benchmark/scipost_fig1/run_fig1_right.jl
julia --project=benchmark benchmark/scipost_fig1/plot_fig1.jl
```

Outputs:

```text
results/fig1_left.csv   timestep sweep for panels (a,b)
results/fig1_right.csv  propagation-order sweep for panels (c,d)
results/environment.txt
results/fig1.pdf
results/fig1.png
```

Defaults: $T=6$, left-column timesteps `0.20,0.10,0.05`, right-column
$\Delta t=0.10$. Smoke run:

```bash
SCIPOST_FIG1_T=1.2 SCIPOST_FIG1_DTS=0.2,0.1 \
  OPENBLAS_NUM_THREADS=1 julia -t auto --project=benchmark \
  benchmark/scipost_fig1/run_fig1_left.jl
```

Environment overrides: `SCIPOST_FIG1_T`, `SCIPOST_FIG1_DTS`,
`SCIPOST_FIG1_RIGHT_DT`, `SCIPOST_FIG1_CUTOFF`. Each final time must be an
integer multiple of every selected timestep.
