# Terminal progress and verbose demos

Run these scripts from a visible terminal:

```bash
julia --project=. scripts/terminal/build_process_tensor.jl
julia --project=. scripts/terminal/create_instruments.jl
julia --project=. scripts/terminal/evaluate_process.jl
julia --project=. scripts/terminal/evolve.jl
julia --project=. scripts/terminal/tebd.jl
julia --project=. scripts/terminal/spinner_demo.jl
```

Each script sets `progress=true, verbose=true` so that transient ProgressMeter
feedback and durable Julia `@info` records are visible together.

`spinner_demo.jl` walks through the header-spinner + child-bar layout used by
`build_process_tensor`. Prefer `julia -t auto --project=.
scripts/terminal/spinner_demo.jl` so the glyph keeps rotating during CPU work.

To emulate a headless job, change the local settings to:

```julia
progress = false
verbose = true
```

For a quiet run, use `progress=false, verbose=false`. Progress bars are
transient and clear when their stage completes; verbose records remain suitable
for redirection, for example `julia --project=. scripts/terminal/evolve.jl 2>&1
| tee run.log`.
