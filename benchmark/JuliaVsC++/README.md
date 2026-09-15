# Julia vs C++ ACE construction

Head-to-head process-tensor **construction** timings and maximum temporal bond
dimensions $D_{\max}$ for ProcessTensors.jl against Moritz Cygorek's C++ ACE
toolkit (`ACE/`, [mcygorek/ACE](https://github.com/mcygorek/ACE)).

This folder records every command used on this machine, from installing Eigen
through the production sweeps. The measured quantity for the Julia–C++ fight is
ACE **process-tensor construction**, not subsequent system contraction. C++ runs
set `dont_propagate true`. Julia runs call `build_process_tensor` and stop there.

Table S.3.1 of the [Nat. Phys. 2022 supplement](https://doi.org/10.1038/s41567-022-01544-9)
reports wall times for **complete examples** (construction + contraction + I/O).
Those numbers and the matching ACE inputs live in the sanity folders, not in
this head-to-head. Use `build_s` for the Julia–C++ comparison.

C++ ACE lives at `ACE/` and is gitignored from the Julia package. It is a
separate clone, not part of ProcessTensors.jl.

## 0. Models

ProcessTensors.jl uses $\hbar=1$, and its spin operators are
$S_\alpha=\sigma_\alpha/2$. C++ ACE interprets Hamiltonians in meV and time
in ps, with `hbar_in_meV_ps = 0.6582119569`. Consequently, a C++ input
$J=1$ evolves at the dimensionless rate $J/\hbar=1.5192674479961275$.
For equal dimensionless dynamics, either set the C++ coupling to
$0.6582119569\,J$ or set the Julia coupling to $J/0.6582119569$.

### Central spin

Hamiltonian, matching Cygorek et al., Nat. Phys. 18, 662 (2022) and
`ACE/examples/NatPhys2022/03_spins/`:

$$
H_S=0,\qquad
H_{B_k}=0,\qquad
H=\sum_{k=1}^{N}\frac{J}{N}
\bigl(S_x s_k^x+S_y s_k^y+S_z s_k^z\bigr).
$$

Fixed production parameters:

| Quantity | Value |
| ---: | ---: |
| $J$ | $1$ |
| final time $JT$ | $20$ |
| timestep $J\Delta t$ | $0.1$ |
| ACE threshold $\varepsilon$ | $10^{-10}$ |
| number of PT cores | $n=T/\Delta t=200$ |
| $N$ | $5,10,25,50,100$ |
| system Trotter | second order |
| mode-combination Trotter | second order |
| C++ flag | `use_symmetric_Trotter true` |
| Julia flags | `sys_alg=Trotter{2}()`, `combine_alg=Trotter{2}()` |
| Julia compression | `:zipup_cpp` (C++ `add_modes` order, ITensors `gesdd`) |

System initial state (all three polarisations):

$$
\rho_S(0)=\lvert+\rangle_x\langle+\rvert
=\tfrac12\bigl(I+\sigma_x\bigr).
$$

In C++ that is

```text
initial  { 0.5 * ( Id_2 + |0><1|_2 + |1><0|_2 ) }
```

Bath initial conditions, one case per polarisation:

| Polarisation | Bath state | C++ knobs |
| ---: | :--- | :--- |
| fully polarised | every spin $\lvert\uparrow\rangle_z$ | `RandomSpin_set_initial_dir 0 0 1` |
| partially polarised | random pure states, Boltzmann-filtered against $\mathbf B=(0,0,20)$ at $T=1/k_B$ | `RandomSpin_B_init 0 0 20` and `RandomSpin_T 11.604522`, same as `ACE/examples/NatPhys2022/03_spins/02_b20_*.param` |
| unpolarised | independent random pure states | omit both; `RandomSpin_seed 1` |

Partial polarisation is still a product of **pure** states, not a mixed Gibbs
state per spin. That is Cygorek's ACE protocol, not a thermal density matrix.

Couplings are $J_k=J/N$:

| $N$ | $J_k$ |
| ---: | ---: |
| 5 | 0.2 |
| 10 | 0.1 |
| 25 | 0.04 |
| 50 | 0.02 |
| 100 | 0.01 |

### Thermal spin-boson (independent boson)

Spectral density and discretisation follow Nat. Phys. 2022 Supplementary
Information, Sec. S.4.D (the harmonic / independent-boson reference used next
to the Morse bath):

$$
J(\omega)
=
\frac{C}{\pi}
\frac{\gamma}{(\omega-\omega_c)^2+\gamma^2},
\qquad
\omega_c=\Omega,\quad
\gamma=0.1\,\Omega.
$$

Modes are the midpoints of $N_E$ equal intervals on
$0\le\omega/\Omega\le 7.5$, and

$$
g_k=\sqrt{J(\omega_k)\,\Delta\omega},\qquad
\Delta\omega=7.5\,\Omega/N_E.
$$

Each oscillator is truncated to $M=5$ Fock levels
($\lvert 0\rangle,\ldots,\lvert 4\rangle$). The SI reference run used
$C=0.1\,\Omega^2$ and $k_B T=0.5\,\Omega$. This sweep keeps that grid and
$M$, and uses the requested production coupling and temperatures:

| Quantity | SI Sec. S.4.D | This sweep |
| ---: | ---: | ---: |
| $C/\Omega^2$ | $0.1$ | $0.2$ |
| $k_B T/\Omega$ | $0.5$ | $0,0.5,1.0,3.0$ |
| $N_E$ | 100 | $5,10,25,50,100$ |
| $M$ | 5 | 5 (fixed for every $T$) |
| $\omega/\Omega$ | $[0,7.5]$ | $[0,7.5]$ |
| $\Omega T$ | (Fig. 5 window) | $8.0$ |
| $\Omega\Delta t$ | | $0.1$ |
| $\varepsilon$ | | $10^{-8}$ |

System Hamiltonian as in SI S.4.D:

$$
H_S=\frac{\Omega}{2}\sigma_x
=\frac{\Omega}{2}\bigl(\lvert e\rangle\langle g\rvert+\lvert g\rangle\langle e\rvert\bigr).
$$

C++:

```text
add_Hamiltonian  {hbar/2.* 1. *(|0><1|_2+|1><0|_2)}
```

Julia (ITensor `Sx = σx/2`):

```julia
H += 1.0, "Sx", 1    # Ω = 1
```

Initial states:

$$
\rho_S(0)=\lvert g\rangle\langle g\rvert=\lvert 0\rangle\langle 0\rvert,
\qquad
\rho_k(0)=\frac{e^{-\hbar\omega_k n/k_B T}}{Z_k}
\Big\lvert_{n=0}^{M-1}.
$$

At $T=0$ every mode is the vacuum. $M$ does **not** change with
temperature; a hotter bath only changes the Gibbs weights on the same five
levels.

Mode Hamiltonian and independent-boson coupling (including the default ACE
polaron shift $g_k^2/\omega_k\lvert e\rangle\langle e\rvert$):

$$
H_{B_k}=\omega_k a_k^\dagger a_k,
\qquad
H_{\mathrm{int},k}
=
g_k(a_k+a_k^\dagger)\lvert e\rangle\langle e\rvert
+\frac{g_k^2}{\omega_k}\lvert e\rangle\langle e\rvert.
$$

## 1. Install the C++ build dependencies

C++ ACE compiles with `g++` and Make. It needs the Eigen headers. This machine
did not have them; Ubuntu provides `libeigen3-dev` 3.4.0.

From anywhere:

```bash
sudo apt install libeigen3-dev
```

Check that the headers are where the Makefile looks:

```bash
ls /usr/include/eigen3/Eigen/Eigen
```

CMake is not required. The Makefile path is enough.

## 2. Compile C++ ACE

The clone is `ACE/` at the repository root. A first `make` compiles every
`src/*.cpp` to `lib/*.o`, links `lib/libACE.so`, then builds `bin/ACE`,
`bin/QUAPI`, `bin/TEMPO`, and the `tools/` utilities (including `PTB_analyze`).

```bash
cd ACE
make
```

This already completed here in Paanini server (about seven minutes). Confirm:

```bash
ls -l ACE/bin/ACE ACE/tools/PTB_analyze
```

Put the binaries on `PATH` for the rest of the session:

```bash
export PATH="$PWD/ACE/bin:$PWD/ACE/tools:$PATH"
```

If you are not in the repository root, use the absolute path instead:

```bash
export PATH="/home/gautham/ProcessTensors.jl/ACE/bin:/home/gautham/ProcessTensors.jl/ACE/tools:$PATH"
```

Smoke-test the binary (should print usage or a missing-driver error, not
`command not found`):

```bash
ACE
```

Rebuild after changing C++ sources with another `cd ACE && make`. Object files
already present are reused.

## 3. Spectral density inside C++ ACE

C++ ACE already implements the SI Lorentzian as `Boson_J_type lorentzian`:

```text
J(ω) = (C / π) γ / ((ω - ω_c)² + γ²)
```

with

```text
Boson_J_type    lorentzian
Boson_J_gamma   0.1
Boson_J_scale   0.2
Boson_J_shift   1.0
```

ACE builds a unit Lorentzian of width `Boson_J_gamma` centred at the origin,
then applies $J_{\mathrm{used}}(\omega)=C\,J_{\mathrm{unit}}(\omega-\omega_c)$
(`Boson_J_scale` $=C/\Omega^2$, `Boson_J_shift` $=\omega_c/\Omega$). The mode
grid is still `Boson_N_modes`, `Boson_omega_min`, and `Boson_omega_max`.
`spinboson.param` already contains these lines.

The SI reference coupling $C=0.1\,\Omega^2$ is the same type with
`Boson_J_scale 0.1`. Julia evaluates the same closed form in `common.jl`;
it never reads a spectral-density file.

## 4. Central spin: C++ ACE

Parameter templates:

```text
benchmark/JuliaVsC++/cpp/central_spin_polarised.param
benchmark/JuliaVsC++/cpp/central_spin_partial.param
benchmark/JuliaVsC++/cpp/central_spin_unpolarised.param
```

Each file already has $J\Delta t=0.1$, $JT=20$, $\varepsilon=10^{-10}$,
`use_symmetric_Trotter true`, and `dont_propagate true`. The sweep overrides
$N$ and $J_k=1/N$ on the command line.

The wrapper writes wall time (`/usr/bin/time`), ACE's `Maxdim` lines, an
optional `PTB_analyze` check, provenance, and CSV rows. Each case is pinned to
`ACE_CPUS` is a pool: unfinished cases run one-per-core from that list, and
the rest wait for a free core. Each running case is pinned with
`OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`.
Completed CSV rows are skipped.

```bash
bash benchmark/JuliaVsC++/run_cpp_central_spin.sh
```

Pin an explicit set of CPUs (unique ids; one case per core):

```bash
ACE_CPUS="0 1 2 3" NS="5 10" POLS="polarised" \
  bash benchmark/JuliaVsC++/run_cpp_central_spin.sh
```

Equivalent manual commands, from the repository root, for every $(N,$
polarisation) pair. Replace `N` and `Jk` from the table above.

Fully polarised, example $N=10$:

```bash
mkdir -p benchmark/JuliaVsC++/results/cpp
/usr/bin/time -f '%e' -o benchmark/JuliaVsC++/results/cpp/central_polarised_N10.time \
  ACE benchmark/JuliaVsC++/cpp/central_spin_polarised.param \
    -RandomSpin_N_modes 10 \
    -RandomSpin_J_max 0.1 \
    -RandomSpin_J_min 0.1 \
    -dont_propagate true \
    -write_PT benchmark/JuliaVsC++/results/cpp/central_polarised_N10.pt \
    -RandomSpin_print_initial benchmark/JuliaVsC++/results/cpp/central_polarised_N10_orientations.txt \
  | tee benchmark/JuliaVsC++/results/cpp/central_polarised_N10.log
```

Partially polarised, example $N=25$:

```bash
/usr/bin/time -f '%e' -o benchmark/JuliaVsC++/results/cpp/central_partial_N25.time \
  ACE benchmark/JuliaVsC++/cpp/central_spin_partial.param \
    -RandomSpin_N_modes 25 \
    -RandomSpin_J_max 0.04 \
    -RandomSpin_J_min 0.04 \
    -dont_propagate true \
    -write_PT benchmark/JuliaVsC++/results/cpp/central_partial_N25.pt \
    -RandomSpin_print_initial benchmark/JuliaVsC++/results/cpp/central_partial_N25_orientations.txt \
  | tee benchmark/JuliaVsC++/results/cpp/central_partial_N25.log
```

Unpolarised, example $N=50$:

```bash
/usr/bin/time -f '%e' -o benchmark/JuliaVsC++/results/cpp/central_unpolarised_N50.time \
  ACE benchmark/JuliaVsC++/cpp/central_spin_unpolarised.param \
    -RandomSpin_N_modes 50 \
    -RandomSpin_J_max 0.02 \
    -RandomSpin_J_min 0.02 \
    -dont_propagate true \
    -write_PT benchmark/JuliaVsC++/results/cpp/central_unpolarised_N50.pt \
    -RandomSpin_print_initial benchmark/JuliaVsC++/results/cpp/central_unpolarised_N50_orientations.txt \
  | tee benchmark/JuliaVsC++/results/cpp/central_unpolarised_N50.log
```

Repeat for $N\in\{5,10,25,50,100\}$ and all three polarisations (fifteen
runs). After each run, read $D_{\max}$ from the log or from the written PT:

```bash
grep 'Maxdim' benchmark/JuliaVsC++/results/cpp/central_polarised_N10.log
PTB_analyze -read_PT benchmark/JuliaVsC++/results/cpp/central_polarised_N10.pt
```

`PTB_analyze` prints `Maxdim <D> at <site>`. ACE construction also prints
`Maxdim at n=...: ... -> D` after each mode join; the last outgoing `D` is
the final $D_{\max}$.

`/usr/bin/time` writes the elapsed seconds (one number) to the `.time` file.
That is the PT-construction wall time, because `dont_propagate` skips
system evolution. ACE itself only prints `runtime for propagation` when
propagation is enabled.

## 5. Central spin: ProcessTensors.jl

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  JULIA_VS_CPP_CPUS="8 9 10 11" \
  julia -t 1 --project=. benchmark/JuliaVsC++/run_julia_central_spin.jl
```

`JULIA_VS_CPP_CPUS` (or `JULIA_SPINBOSON_CPUS`) is the Julia analogue of `ACE_CPUS`. Cases rotate over that list. If a listed core already has a running (`R`) task, the case is moved to the next free core and a line is printed. Omit the variable to leave affinity unset.

Smoke (polarised $N=5$ only):

```bash
JULIA_VS_CPP_SMOKE=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  julia -t 1 --project=. benchmark/JuliaVsC++/run_julia_central_spin.jl
```

The C++-schedule and existing-C++-data comparison can be reproduced without
overwriting the main Julia CSV:

```bash
JULIA_VS_CPP_SMOKE=1 \
  JULIA_VS_CPP_COMPRESSION=zipup_cpp \
  JULIA_VS_CPP_J=1.5192674479961275 \
  JULIA_VS_CPP_OUTPUT=central_spin_zipup_cpp_smoke.csv \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  julia -t 1 --project=. benchmark/JuliaVsC++/run_julia_central_spin.jl
```

The scripts force one OpenBLAS/MKL/OpenMP thread (`JULIA_VS_CPP_BLAS_THREADS`,
default 1) and expect `julia -t 1`. Launch each parallel worker with `taskset`
onto a distinct CPU so two cases never share a core. The script times **only**
`build_process_tensor`: one untimed warmup discards compilation, then
BenchmarkTools (`evals=1`, default `samples=1`) records construction. Bath setup
and C++ orientation parsing sit outside that timer and are stored as `setup_s`.

Partial and unpolarised baths reuse
`results/cpp/central_{partial,unpolarised}_N*_orientations.txt`. Polarised
baths are all $+z$ (the matching C++ dump is used if present). The script
reads those files and writes only
`benchmark/JuliaVsC++/results/julia/central_spin.csv`, or the filename set by
`JULIA_VS_CPP_OUTPUT`.

```julia
build_process_tensor(
    system;
    method=ACE(cutoff=1e-10, maxdim=4096, compression=:zipup_cpp),
    environment=bath,
    dt=0.1,
    nsteps=200,
    alg=Exact(),
    sys_alg=Trotter{2}(),
    combine_alg=Trotter{2}(),
)
```

## 6. Thermal spin-boson: C++ ACE

```bash
bash benchmark/JuliaVsC++/run_cpp_spinboson.sh
```

The runner sweeps every missing pair in
$k_B T/\Omega\in\{0,0.5,1,3\}$ and $N_E\in\{5,10,20,50,100\}$, matching
the Julia grid. It appends to the existing CSV and skips completed
$(T,N_E)$ keys. Each missing case is pinned to a CPU from `ACE_CPUS`.
Extra cases wait until a core in that list is free. Override either sweep:

```bash
ACE_CPUS="0 1 2 3" TEMPS="0.5 1.0" NS="5 10" \
  bash benchmark/JuliaVsC++/run_cpp_spinboson.sh
```

Manual equivalent for one case:

```bash
/usr/bin/time -f '%e' -o benchmark/JuliaVsC++/results/cpp/spinboson_T0.5_N50.time \
  ACE benchmark/JuliaVsC++/cpp/spinboson.param \
    -temperature_unitless 0.5 \
    -Boson_N_modes 50 \
    -dont_propagate true \
    -write_PT benchmark/JuliaVsC++/results/cpp/spinboson_T0.5_N50.pt
```

`temperature_unitless` is $k_B T/\hbar$ in the same units as `Boson_omega_*`.
With $\Omega=1$ this is exactly $k_B T/\Omega$. `Boson_M 5` is fixed.

Read $D_{\max}$:

```bash
grep 'Maxdim' benchmark/JuliaVsC++/results/cpp/spinboson_T0.5.log
PTB_analyze -read_PT benchmark/JuliaVsC++/results/cpp/spinboson_T0.5.pt
```

Optional: dump the actual $\{E_k,g_k\}$ ACE used

```bash
ACE benchmark/JuliaVsC++/cpp/spinboson.param \
  -Boson_print_E_g benchmark/JuliaVsC++/results/cpp/spinboson_E_g.dat \
  -Boson_stop_after_print_E_g true
```

## 7. Thermal spin-boson: ProcessTensors.jl

Run all missing $(T,N_E)$ cases concurrently, one Julia process and one
BLAS/OpenMP thread per exclusive CPU. $N_E\in\{5,10,20,50,100\}$. A CSV
row is skipped only if $N_E$, $\varepsilon$, `:zipup_cpp`, and thread
count all match; blank thread fields are rerun.

```bash
bash benchmark/JuliaVsC++/run_julia_spinboson_parallel.sh
```

Choose an explicit set of dedicated CPUs or a subset of cases with:

```bash
ACE_CPUS="16 17 18 19" \
TEMPS="0 0.5" NS="5 10" \
  bash benchmark/JuliaVsC++/run_julia_spinboson_parallel.sh
```

The serial resumable runner remains available:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  JULIA_VS_CPP_CPUS="8 9 10 11" \
  JULIA_VS_CPP_TEMPS="0,0.5" JULIA_VS_CPP_NS="5,10" \
  julia -t 1 --project=. benchmark/JuliaVsC++/run_julia_spinboson.jl
```

Both runners preserve only rows that match compressor and threads. The
parallel wrapper writes one isolated fragment per case and merges
fragments only after workers finish, so concurrent workers never append
to the same file. Same $N_E$, $M$, $C$, $\omega$ window, $\Delta t$,
$T_{\mathrm{final}}$, $\varepsilon$, and `Trotter{2}()` / `:zipup_cpp`
settings are used on both implementations. Output:

```text
benchmark/JuliaVsC++/results/julia/spinboson.csv
```

## 8. Plot the comparison

After any subset of the benchmark runs has completed, generate the two-panel
runtime versus maximum-bond-dimension comparison with:

```bash
julia --project=. benchmark/JuliaVsC++/plot_results.jl
```

Julia results use filled markers and dashed lines; C++ results use matching
hollow markers and dotted lines. The script skips missing or incomplete CSV
rows. Spin-boson series are grouped by temperature and annotated by $N_E$, so
partially completed sweeps can be plotted. Output:

```text
benchmark/JuliaVsC++/results/julia_vs_cpp.pdf
benchmark/JuliaVsC++/results/julia_vs_cpp.png
```

## 9. What to record

After the four wrappers (or the manual commands) you should have:

```text
benchmark/JuliaVsC++/results/cpp/central_spin.csv
benchmark/JuliaVsC++/results/cpp/spinboson.csv
benchmark/JuliaVsC++/results/julia/central_spin.csv
benchmark/JuliaVsC++/results/julia/spinboson.csv
```

Each result CSV row is one PT construction. Keep the original runtime and
$D_{\max}$ columns for plotting, and use the extra columns for provenance:

| Column | Meaning |
| ---: | :--- |
| `build_s` | PT construction (Julia–C++ fight) |
| `contract_s` | system propagation; 0 under `dont_propagate` / build-only Julia |
| `io_s` | `PTB_analyze` or other I/O charged separately |
| `total_s` | C++: ACE process wall time; Julia: `setup_s + build_s` |

C++ also keeps `.log`, `.time`, `.pt`, and orientation dumps. The `.pt`
binaries are large and gitignored.

Suggested hardware notes: AMD Ryzen Threadripper 7980X. Head-to-head runs use
`julia -t 1` and `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`, with
each parallel case on a distinct `taskset` CPU. C++ ACE without MKL unless you
rebuild. Record Julia, ProcessTensors, ITensors, ITensorMPS, `g++`, and Eigen
versions in the paper methods; the runners print that block at startup.

## 10. Published Table S.3.1 (Cygorek et al., Nat. Phys. 2022)

Table S.3.1 health checks live in `sanity_central_spin/` and
`sanity_spin_boson/`. Each folder has its own `published_runtimes.csv`
and construction-only ACE inputs. `timing_scope` is `complete_example`
for every published row: those times are **not** pure PT-construction
times.

Published central-spin examples use $\Delta t=0.01\,\hbar/J$,
$t_{\mathrm{final}}=20\,\hbar/J$ (2000 steps). This production sweep keeps
$\Delta t=0.1$, $n=200$, and `dont_propagate true`.

Published independent-boson / Morse examples use the recovered input
`01_independent_boson_T0.5_M5_Jscale0.1.param`: $N=101$, $C/\Omega^2=0.1$,
$k_BT=0.5\Omega$, $\Delta t=0.1$, $t_{\mathrm{final}}=10$, $\varepsilon=10^{-7}$.
Do not borrow $\Delta t$ or $\varepsilon$ from another SI figure. This
production sweep uses $C/\Omega^2=0.2$, $N_E\in\{5,10,25,50,100\}$,
$t_{\mathrm{final}}=8$, $\varepsilon=10^{-8}$, so it is also not an exact
Table S.3.1 reproduction.

The $\varepsilon=10^{-16}$ central-spin times exceeded the laptop's 16 GB and
swapped; they are stress tests, not clean calibration targets. The three
recommended C++ health checks (still Table S.3.1 complete-example times, on
an i5-8265U with MKL) are:

| Purpose | Case | Published |
| :--- | :--- | ---: |
| Easy baseline | fully polarised, $N=100$, $\varepsilon=10^{-10}$ | 4 min 09 s |
| Representative difficult | unpolarised, $N=100$, $\varepsilon=10^{-10}$ | 8 min 27 s |
| Compression stress | unpolarised, $N=100$, $\varepsilon=10^{-13}$ | 40 min 52 s |

Beating those times on the Threadripper shows the C++ binary is a healthy
`-O3` build, not that Julia is faster. The defensible speed comparison remains
Julia versus C++ on this server under matched single-core limits.

## 11. Matching caveats

- Fully polarised central spin is a like-for-like microscopic match.
- Unpolarised / partial Julia runs import the C++ `*_orientations.txt`
  dumps, so they compress the same bath realisation as the C++ table.
- Julia `:zipup_cpp` follows C++ `add_modes` (truncate then join) with the
  ITensors default SVD (`gesdd`). C++ ACE uses Eigen JacobiSVD. The library alternative `:canonzip` is a different schedule and must not be mixed into this table.
- `sys_alg=Trotter{2}()` / `use_symmetric_Trotter true` is the
  system–environment splitting. `combine_alg=Trotter{2}()` is Julia's
  symmetric half-step mode join. C++ mode propagators are exact for each
  single mode and then joined sequentially.
- Julia embeds $H_S$ into the PT cores after the bath sweep. That does not
  change $D_{\max}$ and is cheap compared with the joins. C++
  `dont_propagate` builds the environment PT only.
- $M=5$ at $k_B T/\Omega=3$ is a controlled truncation, not a converged
  Hilbert-space cutoff. Keep $M$ fixed so the temperature sweep is fair.
- Table S.3.1 must not be labelled as a build-only time except where the
  archived example itself separated construction from contraction.

## 12. Folder

Published-discretisation sanity (single-thread, serial) lives in
`sanity_central_spin/` and `sanity_spin_boson/`. Construction times:

```bash
bash benchmark/JuliaVsC++/run_sanity_cpp_vs_published.sh
```

C++ vs Julia vs ED on small models:

```bash
bash benchmark/JuliaVsC++/run_sanity_cpp_julia_ed.sh
```

```text
benchmark/JuliaVsC++/
  README.md
  common.jl
  provenance.jl
  run_sanity_cpp_vs_published.sh
  run_sanity_cpp_julia_ed.sh
  run_julia_central_spin.jl
  run_julia_spinboson.jl
  run_julia_spinboson_parallel.sh
  run_cpp_central_spin.sh
  run_cpp_spinboson.sh
  plot_results.jl
  sanity_central_spin/
    published_runtimes.csv
  sanity_spin_boson/
    published_runtimes.csv
  cpp/
    bench_common.sh
    extract_maxdim.sh
    central_spin_polarised.param
    central_spin_partial.param
    central_spin_unpolarised.param
    spinboson.param
  results/
    cpp/
    julia/
```
