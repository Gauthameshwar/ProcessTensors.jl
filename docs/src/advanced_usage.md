# Advanced Usage

This page summarizes the execution-feedback options available for long-running
`ProcessTensors.jl` workflows such as `build_process_tensor`,
`create_instruments`, `evaluate_process`, `evolve`, and `tebd`.

## Progress and verbose output

These workflows support two independent options:

```julia
progress = :auto
verbose = false
```

* `progress` controls transient terminal feedback such as spinners and progress
  bars.
* `verbose` controls persistent informational logs that remain after the
  calculation finishes.

| `progress` | `verbose` | Behaviour                                               | Recommended use                                                     |
| ---------- | --------: | ------------------------------------------------------- | ------------------------------------------------------------------- |
| `false`    |   `false` | No live display or informational logs                   | Benchmarks, tests, parameter sweeps, and performance-sensitive runs |
| `false`    |    `true` | Persistent milestone logs without animation             | Remote servers, HPC jobs, batch scripts, and redirected logs        |
| `true`     |   `false` | Transient spinner and progress bars                     | Interactive local runs, tutorials, and demonstrations               |
| `true`     |    `true` | Live progress together with persistent logs             | Local debugging and development                                     |
| `:auto`    |   `false` | Enable progress only in a suitable interactive terminal | Recommended default                                                 |

A normal local run can use the defaults:

```julia
pt = build_process_tensor(
    system;
    environment=bath,
    dt=0.05,
    nsteps=100,
)
```

For a remote or headless calculation:

```julia
pt = build_process_tensor(
    system;
    environment=bath,
    dt=0.05,
    nsteps=100,
    progress=false,
    verbose=true,
)
```

For a silent, performance-oriented run:

```julia
pt = build_process_tensor(
    system;
    environment=bath,
    dt=0.05,
    nsteps=100,
    progress=false,
    verbose=false,
)
```

With `verbose=true`, the package records major algorithmic milestones without
printing one message per timestep. Warnings and errors remain persistent
regardless of the values of `progress` and `verbose`.

Runnable terminal demonstrations are available under `scripts/terminal/` in
the package repository.

## Threading and dynamic progress display

When progress is enabled, a long workflow may display a transient two-line
status:

```text
◓ Building process tensor — assembling cores    Time: 0:00:08
43%|██████████            | 43/100  t=2.15  χ=128
```

The first line shows the current stage and elapsed time. The second may show a
known-length progress bar together with compact quantities already available
to the algorithm, such as simulation time or maximum bond dimension.

The display is erased when the operation finishes, after which Julia handles
the returned object normally.

The spinner uses only Julia execution threads that were available when Julia
started. It prefers an interactive thread when one exists (via the julia call `julia -t 3,1 <script>.jl`), otherwise another
available Julia thread or a cooperative task.

On a single-threaded Julia session, the spinner updates whenever the numerical
calculation yields control. During a long non-yielding operation, the latest
spinner frame may remain visible until that operation finishes. This affects
only the smoothness of the animation, not the result.

The progress backend does not change BLAS thread counts, launch external
processes, alter numerical algorithms, or create additional Julia threads at
runtime.

No special threading setup is required. Users already running Julia with
multiple threads may see smoother animation:

```text
julia --threads=auto --project=. script.jl
```

Use `progress=false` when all available resources should be devoted to the
scientific calculation.
