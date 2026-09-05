# SciPost Figure 2: ACE temporal bond dimensions

This folder measures how ACE `cutoff` and timestep reshape the temporal bond
structure of a process-tensor MPO. Data generation and plotting are separate:
`run_fig2.jl` builds process tensors and writes CSV files, while
`plot_fig2.jl` only reads saved data.

## Unpolarised central-spin model

We set \(\hbar=1\) and use ITensor operators
\(S_\alpha=\sigma_\alpha/2\). The system and bath spins have no free
Hamiltonians:

\[
H_S=H_{B_k}=0,\qquad
H=\sum_{k=1}^{N}\frac{J}{N}\,
\left(S_xs_k^x+S_ys_k^y+S_zs_k^z\right).
\]

The central spin starts along \(+x\), so
\(\langle S_x(0)\rangle=1/2\). Each bath spin starts in an independently
drawn pure state,

\[
|\mathbf n_k\rangle=
\cos(\theta_k/2)|\uparrow\rangle+
e^{i\phi_k}\sin(\theta_k/2)|\downarrow\rangle,
\]

uniform on the Bloch sphere. This is the \(b=0\) unpolarised ensemble used in
the difficult central-spin case.

The default realization uses `Random.Xoshiro(20260905)`. The generated
angles and Bloch vectors are archived in `results/bath_orientations.csv` and
then read back to construct the bath. Every timestep and cutoff therefore
uses precisely the same realization. The seed and orientation-file SHA-256
are recorded in `results/environment.txt`.

## Measured quantities

For a process tensor with \(N_t\) temporal cores, \(D_k\) is the dimension of
the internal link between cores \(Q^{[k]}\) and \(Q^{[k+1]}\). Its physical
cut time is \(t_k=k\Delta t\). The profile CSV includes the unity boundary
bonds at \(t=0\) and \(t=T\); \(D_{\max}\) is computed from internal links
only.

- Figure 2(a): \(D_k(t_k)\) at \(\Delta t_{\rm ref}=0.10\).
- Figure 2(b): \(D_{\max}\) across timestep and cutoff.

Defaults are \(N=50\), \(J=1\), \(T=21\) (the smallest \(T\ge 20\) that is
an integer multiple of every timestep, including \(1.5\)),
\(\Delta t\in\{1.5,0.50,0.20,0.10,0.05\}\), and
\(\epsilon_{\rm SVD}\in\{10^{-6},10^{-8},10^{-10},10^{-12}\}\).
All builds use `compression=:canonzip`, `alg=Exact()`,
`sys_alg=Trotter{2}()`, `combine_alg=Trotter{2}()`, and `maxdim=4096`.
An open marker denotes a point that reached `maxdim`.

## Generate data and figure

From the repository root:

```bash
julia -t auto --project=. benchmark/scipost_fig2/run_fig2.jl
julia --project=. benchmark/scipost_fig2/plot_fig2.jl
```

BLAS is not restricted by the benchmark command. Its actual thread count and
configuration are recorded as provenance.

Outputs:

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
julia -t auto --project=. benchmark/scipost_fig2/run_fig2.jl
```

Environment overrides are `SCIPOST_FIG2_SEED`, `SCIPOST_FIG2_N`,
`SCIPOST_FIG2_J`, `SCIPOST_FIG2_T`, `SCIPOST_FIG2_DT_REF`,
`SCIPOST_FIG2_DTS`, `SCIPOST_FIG2_CUTOFFS`, and
`SCIPOST_FIG2_MAXDIM`. The final time must be an integer multiple of each
timestep.
