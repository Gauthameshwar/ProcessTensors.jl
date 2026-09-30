# Benchmarks

Scripts and figures live in five folders:

- [`benchmark/ace_compressors/`](ace_compressors/) — ACE zipup_cpp and canonzip construction ([`README.md`](ace_compressors/README.md))
- [`benchmark/evolve_contractors/`](evolve_contractors/) — `:evaluate` vs `:closures` evolve
- [`benchmark/scipost_fig1/`](scipost_fig1/) — ED, Dense PT, and ACE accuracy figure ([`README.md`](scipost_fig1/README.md))
- [`benchmark/scipost_fig2/`](scipost_fig2/) — ACE cutoff, timestep, and temporal bonds ([`README.md`](scipost_fig2/README.md))
- [`benchmark/JuliaVsC++/`](JuliaVsC++/) — ProcessTensors.jl vs Cygorek C++ ACE ([`README.md`](JuliaVsC++/README.md))

All of these scripts use the environment defined in
[`benchmark/Project.toml`](Project.toml). Run them as
`julia --project=benchmark …` from the repository root. The first load
instantiates from [`benchmark/Manifest.toml`](Manifest.toml); parallel
workers should set `SKIP_INSTANTIATE=1`. CSV and figure output is written
next to the owning scripts:

```text
benchmark/ace_compressors/results/
benchmark/evolve_contractors/results/
benchmark/scipost_fig1/results/
benchmark/scipost_fig2/results/
```

## SciPost Figure 1

Generates the four-panel correctness and time-discretisation figure for the
four-mode Sz⊗Sz spin bath used in
[`scripts/pt_tfim_multimode.jl`](../scripts/pt_tfim_multimode.jl). Direct
full-system Hilbert-space ED is the discretisation-free reference. `Dense()`
is the exact (uncompressed) PT constructor, while ACE uses sequential mode
joining and temporal compression. The plotted observable is Pauli
$\langle\sigma\_y\rangle$.

The two run scripts save their data independently. A single plotter reads both
CSV files and assembles the complete 2×2 figure:

```bash
OPENBLAS_NUM_THREADS=1 julia -t auto --project=benchmark benchmark/scipost_fig1/run_fig1_left.jl
OPENBLAS_NUM_THREADS=1 julia -t auto --project=benchmark benchmark/scipost_fig1/run_fig1_right.jl
julia --project=benchmark benchmark/scipost_fig1/plot_fig1.jl
```

See [`benchmark/scipost_fig1/README.md`](scipost_fig1/README.md) for the
Hamiltonian, parameter table, output files, and smoke-run overrides.

## SciPost Figure 2

Measures the temporal-memory bond profile of ACE process tensors for the
unpolarised central-spin model. A fixed `Xoshiro` seed generates one archived
set of random pure bath-spin orientations, and every timestep and cutoff uses
that same realization.

```bash
julia -t auto --project=benchmark benchmark/scipost_fig2/run_fig2.jl
julia --project=benchmark benchmark/scipost_fig2/plot_fig2.jl
```

Panel (a) shows the complete $D\_k$ profile at one reference timestep.
Panel (b) shows $D\_{\max}$ across timestep and cutoff. Figure 2 leaves BLAS
unrestricted and records its actual thread count. See
[`benchmark/scipost_fig2/README.md`](scipost_fig2/README.md) for the model,
controls, outputs, and reproducibility metadata.

## ACE compressors

Compares the ACE join/truncation schedules:

```julia
ACE(; cutoff=1e-10, maxdim=typemax(Int), compression=:zipup_cpp)
```

- `:zipup_cpp` (library default): C++ ACE join/truncate order with the ITensors default SVD (`gesdd`), then sweep backward.
- `:canonzip`: join one complete mode without truncation, fuse temporal links, move the orthogonality center to the final time, then one right-to-left ACE relative-SVD sweep.

Use the **same cutoff** for all strategies. Do not retune $\varepsilon$ per algorithm.

```text
ace_compressors/ace_compression_benchmark.jl
ace_compressors/ace_central_spin_scaling.jl
ace_compressors/ace_thread_scaling.jl
ace_compressors/run_ace_thread_scaling.sh
ace_compressors/plot_ace_compression.jl
```

[`ace_compression_benchmark.jl`](ace_compressors/ace_compression_benchmark.jl)
compares zipup_cpp and canonzip across several SVD cutoffs using an
eight-spin environment. The plotted accuracy is the evolved trajectory error
$\max\_k\|\rho\_k-\rho\_k^{\mathrm{ref}}\|\_F$
against a canonzip reference at $\varepsilon=10^{-13}$. zipup_cpp is also
run once at that same cutoff. Timing uses the median BenchmarkTools
sample; the memory axis is total allocated memory, not peak RSS. Trace,
Hermiticity, and positivity diagnostics are stored as `eps_trace`, `eps_H`,
and `eps_pos`.
The $N$-scaling sweep uses Cygorek's unpolarised ($b=0$) central-spin
model with the same $Xoshiro(20260905)$ Bloch protocol as Figure 2.
Defaults are $N\in\{1,5,10,20,30,40,50,70\}$, $T=4$, $\Delta t=0.05$,
and $\varepsilon=10^{-10}$. See
[`benchmark/ace_compressors/README.md`](ace_compressors/README.md).
[`plot_ace_compression.jl`](ace_compressors/plot_ace_compression.jl)
writes one $2\times 3$ figure
`ace_compressors/results/ace_compression.{png,pdf}`:
(a–c) median time, allocated memory, and $\chi\_{\max}$ versus $N$;
(d–e) trajectory error versus time and memory on the eight-spin cutoff sweep;
(f) log build time versus BLAS threads for one easy polarised central-spin
PT (solid) and one hard spin-boson PT (dashed).
Do not `taskset` the thread sweep to a single core.

```bash
julia -t auto --project=benchmark benchmark/ace_compressors/ace_compression_benchmark.jl
julia -t auto --project=benchmark benchmark/ace_compressors/ace_central_spin_scaling.jl
bash benchmark/ace_compressors/run_ace_thread_scaling.sh
julia --project=benchmark benchmark/ace_compressors/plot_ace_compression.jl
```

## Evolve contractors

Times `:evaluate` vs `:closures` on synthetic random process tensors
with prescribed `nsteps` and χ (no ACE construction).

```bash
julia -t auto --project=benchmark benchmark/evolve_contractors/evolve_contraction.jl
julia --project=benchmark benchmark/evolve_contractors/plot_evolve_contraction.jl
```

Runtime and memory use BenchmarkTools (`@benchmarkable` + `run`; `evals=1`).
The first build of each case is used for physics (χ, trajectory error). Timing
comes from later samples after warmup. CSV `t_build` is the **minimum** sample
time in seconds; `t_median_s`, `t_mean_s`, `allocated_bytes`, `allocs`, and
`nsamples` are also written.

## Julia vs C++ ACE

Head-to-head ACE **process-tensor construction** against the cloned C++ ACE
toolkit in `ACE/`. Documents every command from
`sudo apt install libeigen3-dev` through the production sweeps. See
[`benchmark/JuliaVsC++/README.md`](JuliaVsC++/README.md) for Hamiltonians,
initial conditions, and the exact C++ / Julia launch lines.

Central spin: $J=1$, $JT=20$, $J\Delta t=0.1$, $\varepsilon=10^{-10}$,
second-order Trotter, $N\_E\in\{5,10,25,50,100\}$, fully / partially / unpolarised
baths.

Lorentzian spin-boson (Nat. Phys. 2022 SI Sec. S.4.D grid, production
$C/\Omega^2=0.2$): $M=5$, $\Omega T=8$, $\Omega\Delta t=0.1$,
$\varepsilon=10^{-8}$, $N\_E\in\{5,10,20,50,100\}$,
$k\_B T/\Omega\in\{0,0.5,1.0,3.0\}$.

```bash
sudo apt install libeigen3-dev
cd ACE && make && cd ..
export PATH="$PWD/ACE/bin:$PWD/ACE/tools:$PATH"
bash benchmark/JuliaVsC++/run_cpp_central_spin.sh
bash benchmark/JuliaVsC++/run_cpp_spinboson.sh
JULIA_VS_CPP_SMOKE=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  julia -t 1 --project=benchmark benchmark/JuliaVsC++/run_julia_central_spin.jl
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  julia -t 1 --project=benchmark benchmark/JuliaVsC++/run_julia_central_spin.jl
bash benchmark/JuliaVsC++/run_julia_spinboson_parallel.sh
```

Head-to-head runs pin each case to one exclusive CPU with one OpenMP/MKL/OpenBLAS
thread and `julia -t 1`. Construction time is BenchmarkTools after an untimed
warmup, so compilation is not in the CSV. Table S.3.1 published times are
complete-example wall times; see `JuliaVsC++/published_runtimes.csv` and
[benchmark/JuliaVsC++/README.md](benchmark/JuliaVsC++/README.md). Partial and unpolarised Julia cases parse the C++
orientation dumps and reconstructs its modes identical to it. Output goes to `benchmark/JuliaVsC++/results/julia/` only.
