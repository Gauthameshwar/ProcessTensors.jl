# Benchmarks

Scripts and figures live in two folders:

```text
benchmark/ace_compressors/     ACE zip-up vs canonzip construction
benchmark/evolve_contractors/  :evaluate vs :closures evolve
```

Shared local environments stay at `benchmark/.bench_env/` (BenchmarkTools) and
`benchmark/.plot_env/` (CairoMakie). They are not added to the package
`Project.toml`. CSV and figure output is written next to the owning scripts:

```text
benchmark/ace_compressors/results/
benchmark/evolve_contractors/results/
```

## ACE compressors

Compares the two ACE join/truncation schedules:

```julia
ACE(; cutoff=1e-10, maxdim=typemax(Int), compression=:canonzip)
```

- `:canonzip` (library default): join one complete mode without truncation, fuse temporal links, move the orthogonality center to the final time, then one right-to-left ACE relative-SVD sweep.
- `:zipup`: truncate during the forward join, then sweep backward.

Use the **same cutoff** for both strategies. Do not retune \(\varepsilon\) per algorithm.

```text
ace_compressors/ace_compression_sanity.jl
ace_compressors/ace_compression_benchmark.jl
ace_compressors/ace_central_spin_scaling.jl
ace_compressors/plot_ace_compression.jl
```

`ace_compression_benchmark.jl` compares zip-up and canonzip across several SVD
cutoffs using an eight-spin environment, measuring process accuracy, bond
dimension, runtime, and memory allocation. The error-versus-cost plots can be
read as a tradeoff frontier; that interpretation does not need to be in the
filename.

```bash
julia -t auto --project=. benchmark/ace_compressors/ace_compression_sanity.jl
julia -t auto --project=. benchmark/ace_compressors/ace_compression_benchmark.jl
julia -t auto --project=. benchmark/ace_compressors/ace_central_spin_scaling.jl
julia -t auto --project=. benchmark/ace_compressors/plot_ace_compression.jl
```

Optional larger central-spin sweep:

```bash
ACE_RUN_LARGE=true julia -t auto --project=. benchmark/ace_compressors/ace_central_spin_scaling.jl
```

```bash
ACE_BENCH_SAMPLES=5 ACE_BENCH_SECONDS=600 julia -t auto --project=. benchmark/ace_compressors/ace_compression_sanity.jl
```

## Evolve contractors

Times `:evaluate` vs `:closures` on synthetic random process tensors
with prescribed `nsteps` and χ (no ACE construction).

```bash
julia -t auto --project=. benchmark/evolve_contractors/evolve_contraction.jl
julia --project=. benchmark/evolve_contractors/plot_evolve_contraction.jl
```

Runtime and memory use BenchmarkTools (`@benchmarkable` + `run`; `evals=1`).
The first build of each case is used for physics (χ, trajectory error). Timing
comes from later samples after warmup. CSV `t_build` is the **minimum** sample
time in seconds; `t_median_s`, `t_mean_s`, `allocated_bytes`, `allocs`, and
`nsamples` are also written.
