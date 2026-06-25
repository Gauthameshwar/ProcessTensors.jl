```@meta
CurrentModule = ProcessTensors
```

# Examples

These examples show the ProcessTensors layer on top of ITensorMPS objects:
Hilbert-space states, density operators, Liouville vectorization, and
process-tensor workflows.

## Hilbert To Liouville

```julia
using ProcessTensors

sites = siteinds("S=1/2", 4)

ψ = random_mps(sites; linkdims=4)       # MPS{Hilbert}
ρ = to_dm(ψ)                            # MPO{Hilbert}
s_L = liouv_sites(sites)
ρL = to_liouville(ρ; sites=s_L)         # MPS{Liouville}
```

`s_L` should be reused when constructing Liouville MPOs or instruments that
contract with `ρL`.

## Liouvillian MPO

```julia
using ProcessTensors

sites = siteinds("S=1/2", 1)
s_L = liouv_sites(sites)

H = OpSum()
H += 1.0, "Sz", 1

jumps = [(0.1, "S-", 1)]
L = MPO_Liouville(H, s_L; jump_ops=jumps)
```

`L` is an `MPO{Liouville}` acting on vectorized density matrices built on `s_L`.

## Process Tensor Workflow

```julia
using ProcessTensors

sites = siteinds("S=1/2", 1)
s_L = liouv_sites(sites)

H = OpSum()
H += 1.0, "Sz", 1
system = spin_system(s_L, H)

pt = build_process_tensor(system, only(system.sites); dt=0.1, nsteps=4)

ρ0 = to_dm(random_mps(sites; linkdims=1))
seq = InstrumentSeq(default=IdentityOperation(), nsteps=pt.nsteps)
add!(seq, StatePreparation(ρ0), 0)

trajectory = evolve(pt, seq)
```

The returned `trajectory` contains Liouville and Hilbert-space reduced states at
the process-tensor output times.

