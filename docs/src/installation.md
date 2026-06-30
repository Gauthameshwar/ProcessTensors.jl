# Installation

`ProcessTensors.jl` requires **Julia 1.10 or later**. The package is under active development and is installed from GitHub for v0.1.0; it is not yet on the General Registry.

## Install the package

From the Julia REPL:

```julia
using Pkg
Pkg.add(url="https://github.com/Gauthameshwar/ProcessTensors.jl")
```

Then load it in a session:

```julia
using ITensors
using ProcessTensors
```

`ProcessTensors.jl` builds on [`ITensors.jl`](https://docs.itensor.org/ITensors/stable/) and [`ITensorMPS.jl`](https://docs.itensor.org/ITensorMPS/stable/). Those packages are installed automatically as dependencies.

## Development install

To work on a local checkout:

```julia
using Pkg
Pkg.develop(path="/path/to/ProcessTensors.jl")
```

Or, from the package root in the Pkg REPL (`]`):

```text
dev .
```

Run the test suite from the package root:

```julia
using Pkg
Pkg.test("ProcessTensors")
```

## Build the documentation locally

```julia
using Pkg
Pkg.activate("docs")
Pkg.instantiate()
```

Then:

```julia
include("docs/make.jl")
```

Or from the shell:

```text
julia --project=docs docs/make.jl
```

## New to Julia?

If you are new to Julia, these resources are a good starting point:


- [Install Julia](https://julialang.org/install/) — official installation instructions.
- [Julia manual: Getting started](https://docs.julialang.org/en/v1/manual/getting-started/) — running Julia, using the REPL, and getting help.
- [Pkg.jl: Getting started](https://pkgdocs.julialang.org/v1/getting-started/) — installing packages and working with Julia projects.

For tensor-network work in Julia, also skim the [ITensors.jl](https://docs.itensor.org/ITensors/stable/) and [ITensorMPS.jl](https://docs.itensor.org/ITensorMPS/stable/) docs. Much of the syntax in `ProcessTensors.jl` follows those conventions.


!!! tip "Search Julia help from the REPL"
    At the `julia>` prompt, prefix a name with `?` to open its docstring:

    ```julia
    julia> ?to_liouville
    ```

    ```julia
    julia> ?ProcessTensors.build_process_tensor
    ```

    For a broader search across installed packages, use `??`:

    ```julia
    julia> ?? "liouville"
    ```

    This is often the fastest way to check argument order, keyword names, and which module exports a function you have seen in a tutorial.