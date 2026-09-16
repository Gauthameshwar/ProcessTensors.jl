# SciPost Figure 2: ACE temporal bond dimensions

This folder measures how ACE `cutoff` and timestep reshape the temporal bond
structure of a process-tensor MPO. Data generation and plotting are separate:
- [`run_fig2.jl`](run_fig2.jl) builds process tensors and writes CSV files, while
- [`plot_fig2.jl`](plot_fig2.jl) only reads saved data. Shared helpers live in
- [`common.jl`](common.jl).

## Unpolarised central-spin model

We set $\hbar=1$ and use ITensor operators
$S\_\alpha=\sigma\_\alpha/2$. The system and bath spins have no free
Hamiltonians:

$$
H\_S=H\_{B\_k}=0,\qquad
H=\sum\_{k=1}^{N}\frac{J}{N}\,
\left(S\_xs\_k^x+S\_ys\_k^y+S\_zs\_k^z\right).
$$

The central spin starts along $+x$, so
$\langle S\_x(0)\rangle=1/2$. Each bath spin starts in an independently
drawn pure state,

$$
|\mathbf n\_k\rangle=
\cos(\theta\_k/2)|\uparrow\rangle+
e^{i\phi\_k}\sin(\theta\_k/2)|\downarrow\rangle,
$$

uniform on the Bloch sphere. This is the $b=0$ unpolarised ensemble used in
the difficult central-spin case.

The default realization uses `Random.Xoshiro(20260905)`. The generated
angles and Bloch vectors are archived in
[`results/bath_orientations.csv`](results/bath_orientations.csv) and
then read back to construct the bath. Every timestep and cutoff therefore
uses precisely the same realization. The seed and orientation-file SHA-256
are recorded in [`results/environment.txt`](results/environment.txt).

## Measured quantities

For a process tensor with $N\_t$ temporal cores, $D\_k$ is the dimension of
the internal link between cores $Q^{[k]}$ and $Q^{[k+1]}$. Its physical
cut time is $t\_k=k\Delta t$. The profile CSV includes the unity boundary
bonds at $t=0$ and $t=T$; $D\_{\max}$ is computed from internal links
only.

- Figure 2(a): $D\_k(t\_k)$ at $\Delta t\_{\rm ref}=0.10$.
- Figure 2(b): $D\_{\max}$ across timestep and cutoff.

Defaults are $N=50$, $J=1$, $T=21$ (the smallest $T\ge 20$ that is
an integer multiple of every timestep, including $1.5$),
$\Delta t\in\{1.5,0.50,0.20,0.10,0.05\}$, and
$\epsilon\_{\rm SVD}\in\{10^{-6},10^{-8},10^{-10},10^{-12}\}$.
All builds use `compression=:canonzip`, `alg=Exact()`,
`sys_alg=Trotter{2}()`, `combine_alg=Trotter{2}()`, and `maxdim=4096`.
An open marker denotes a point that reached `maxdim`.

## Generate data and figure

From the repository root:

```bash
julia -t auto --project=benchmark benchmark/scipost_fig2/run_fig2.jl
julia --project=benchmark benchmark/scipost_fig2/plot_fig2.jl
```

BLAS is not restricted by the benchmark command. Its actual thread count and
configuration are recorded as provenance.

Outputs under [`results/`](results/):

```text
results/bath_orientations.csv
results/fig2_profiles.csv
results/fig2_dmax.csv
results/environment.txt
results/fig2.pdf
results/fig2.png
```

Smoke run:

```bash
SCIPOST_FIG2_N=4 SCIPOST_FIG2_T=1.2 \
SCIPOST_FIG2_DTS=0.2,0.1 SCIPOST_FIG2_DT_REF=0.1 \
SCIPOST_FIG2_CUTOFFS=1e-6,1e-8 SCIPOST_FIG2_MAXDIM=256 \
julia -t auto --project=benchmark benchmark/scipost_fig2/run_fig2.jl
```

Environment overrides are `SCIPOST_FIG2_SEED`, `SCIPOST_FIG2_N`,
`SCIPOST_FIG2_J`, `SCIPOST_FIG2_T`, `SCIPOST_FIG2_DT_REF`,
`SCIPOST_FIG2_DTS`, `SCIPOST_FIG2_CUTOFFS`, and
`SCIPOST_FIG2_MAXDIM`. The final time must be an integer multiple of each
timestep.
