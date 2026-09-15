# ACE compressors

This folder compares the ACE join/truncation schedules on the same cutoff:

```julia
ACE(; cutoff=1e-10, maxdim=typemax(Int), compression=:zipup_cpp)
```

- `:zipup_cpp` (library default): C++ ACE join/truncate order (truncate the current core before joining the next timestep), then sweep backward. Singular values use the ITensors default SVD (`gesdd`).
- `:canonzip`: join one complete mode without truncation, fuse temporal links, move the orthogonality center to the final time, then one right-to-left ACE relative-SVD sweep.

Do not retune $\varepsilon$ per algorithm.

## Scripts

```text
ace_compression_common.jl
ace_compression_benchmark.jl
ace_central_spin_scaling.jl
ace_thread_scaling.jl
run_ace_thread_scaling.sh
plot_ace_compression.jl
```

`ace_compression_benchmark.jl` compares zipup_cpp and canonzip across
several SVD cutoffs on a heterogeneous eight-spin environment.
Accuracy is the evolved trajectory error
$\max_k\|\rho_k-\rho_k^{\mathrm{ref}}\|_F$
against a canonzip reference at $\varepsilon=10^{-13}$. zipup_cpp is also
run once at that same cutoff. Timing is the median BenchmarkTools
sample; memory is total allocated memory, not peak RSS. Diagnostics are stored
as `eps_trace`, `eps_H`, and `eps_pos`.

## Unpolarised central-spin scaling

`ace_central_spin_scaling.jl` uses the same $b=0$ unpolarised model as SciPost
Figure 2. With $\hbar=1$ and ITensor operators $S_\alpha=\sigma_\alpha/2$,

$$
H_S=H_{B_k}=0,\qquad
H=\sum_{k=1}^{N}\frac{J}{N}\,
\left(S_xs_k^x+S_ys_k^y+S_zs_k^z\right).
$$

The central spin starts along $+x$. Each bath spin starts in an independently
drawn pure state, uniform on the Bloch sphere. One realization is generated
with `Random.Xoshiro(20260905)` for $N_{\max}=70$ and archived in
`results/bath_orientations.csv`. Each smaller $N$ reuses the first $N$
orientations, so the baths are nested.

Defaults: $J=1$, $T=4$, $\Delta t=0.05$ (`nsteps = T/\Delta t + 1`),
$\varepsilon=10^{-10}$, and
$N\in\{1,5,10,20,30,40,50,70\}$. Both strategies are timed at this cutoff.

## Combined figure

`plot_ace_compression.jl` writes one $2\times 3$ figure
`results/ace_compression.{png,pdf}`. Leftover `:zipup` CSV rows are ignored.
Three experiments share the figure:

- (a–c) unpolarised central-spin scaling versus $N$: median time, allocated
  memory, then $\chi_{\max}$. Color is the compressor.
- (d–e) eight-spin cutoff sweep: trajectory error versus time and versus
  allocated memory. Marker shape is $\varepsilon$.
- (f) BLAS-thread scaling of two *fixed* process tensors. Color is the
  compressor; solid is the easy polarised central spin, dashed is the hard
  shortened spin-boson. The $y$ axis is log build time. $\chi$ is annotated
  in the plotter log, not as a second axis.

No axes are shared. Do not `taskset` the thread sweep to a single core.

## BLAS-thread scaling

`run_ace_thread_scaling.sh` launches one `julia -t 1` process per thread
count in $\{1,2,4,8,16,32\}$ so OpenBLAS/MKL see `*_NUM_THREADS` before
startup. The easy case is polarised $N=5$, $\Delta t=0.1$, $t_{\mathrm{final}}=20$,
$\varepsilon=10^{-10}$. The hard case is a Lorentzian independent-boson bath
($C/\Omega^2=0.2$, $N_E=8$, $M=5$, $k_BT/\Omega=3$, $\Delta t=0.1$,
$t_{\mathrm{final}}=6$, $\varepsilon=10^{-8}$). One timed sample after
warmup. Output:
`results/ace_thread_scaling.csv`.

## Generate data and figure

From the repository root:

```bash
taskset -c 0 env OPENBLAS_NUM_THREADS=1 ACE_BENCH_SAMPLES=7 \
  julia -t 1 --project=. benchmark/ace_compressors/ace_compression_benchmark.jl
taskset -c 0 env OPENBLAS_NUM_THREADS=1 ACE_BENCH_SAMPLES=7 \
  julia -t 1 --project=. benchmark/ace_compressors/ace_central_spin_scaling.jl
bash benchmark/ace_compressors/run_ace_thread_scaling.sh
julia --project=. benchmark/ace_compressors/plot_ace_compression.jl
```

```bash
ACE_THREAD_SMOKE=1 ACE_BLAS_THREADS=1 julia -t 1 --project=. \
  benchmark/ace_compressors/ace_thread_scaling.jl
```
