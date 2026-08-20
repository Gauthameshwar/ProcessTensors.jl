# ACE compression benchmarks

This folder compares the two ACE join/truncation schedules:

```julia
ACE(; cutoff=1e-10, maxdim=typemax(Int), compression=:canonzip)
```

- `:canonzip` (library default): join one complete mode without truncation, fuse temporal links, move the orthogonality center to the final time, then one right-to-left ACE relative-SVD sweep.
- `:zipup`: truncate during the forward join, then sweep backward.

Use the **same cutoff** for both strategies. Do not retune \(\varepsilon\) per algorithm.

CSV and figure output:

```text
benchmark/results/
```

## Scripts

```text
ace_compression_sanity.jl
ace_compression_benchmark.jl
ace_central_spin_scaling.jl
```

`ace_compression_benchmark.jl` compares zip-up and canonzip across several SVD
cutoffs using an eight-spin environment, measuring process accuracy, bond
dimension, runtime, and memory allocation. The error-versus-cost plots can be
read as a tradeoff frontier; that interpretation does not need to be in the
filename.

## Run order

From the repository root:

```bash
julia -t auto --project=. benchmark/ace_compression_sanity.jl
julia -t auto --project=. benchmark/ace_compression_benchmark.jl
julia -t auto --project=. benchmark/ace_central_spin_scaling.jl
julia -t auto --project=. benchmark/plot_ace_compression.jl
```

Optional larger central-spin sweep:

```bash
ACE_RUN_LARGE=true julia -t auto --project=. benchmark/ace_central_spin_scaling.jl
```

Runtime and memory use BenchmarkTools (`@benchmarkable` + `run`; `evals=1`).
The first build of each case is used for physics (χ, trajectory error). Timing
comes from later samples after warmup. CSV `t_build` is the **minimum** sample
time in seconds; `t_median_s`, `t_mean_s`, `allocated_bytes`, `allocs`, and
`nsamples` are also written.

BenchmarkTools is loaded from `benchmark/.bench_env/` and is not added to the
package `Project.toml`. The plot script provisions CairoMakie in
`benchmark/.plot_env/`.

```bash
ACE_BENCH_SAMPLES=5 ACE_BENCH_SECONDS=600 julia -t auto --project=. benchmark/ace_compression_sanity.jl
```
