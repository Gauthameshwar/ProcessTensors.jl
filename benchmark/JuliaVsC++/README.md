# Julia vs C++ ACE construction

Head-to-head **process-tensor construction** time and $D\_{\max}$ for
ProcessTensors.jl versus [Cygorek C++ ACE](https://github.com/mcygorek/ACE).
C++ uses `dont_propagate true`. Julia times only `build_process_tensor`.
The comparison column is `build_s`.

C++ ACE is a **separate clone** at [`ACE/`](../../ACE/) (gitignored). Table
S.3.1 of the [Nat. Phys. 2022 supplement](https://doi.org/10.1038/s41567-022-01544-9)
is a complete-example wall time (build + contract + I/O). Checks on the physical validity of 
the process tensors constructed through C++ and Julia live
in [`sanity_central_spin/`](sanity_central_spin/) and
[`sanity_spin_boson/`](sanity_spin_boson/).

C++ ACE uses meV and ps with `hbar_in_meV_ps = 0.6582119569`. 
Dimensionless rates match if C++ $J$ is scaled by $\hbar$ or Julia $J$ by $1/\hbar$.
Spin-boson already matches at $\Omega=1$ because C++ writes `{hbar/2.* 1. * σx}`.

## Models

### Central spin

Hamiltonian, matching Cygorek et al., Nat. Phys. 18, 662 (2022) and
`ACE/examples/NatPhys2022/03_spins/`:

$$
H\_S=H\_{B\_k}=0,\qquad
H=\sum\_{k=1}^{N}\frac{J}{N}
\bigl(S\_x s\_k^x+S\_y s\_k^y+S\_z s\_k^z\bigr).
$$

Production: C++ $J=1\,\mathrm{meV}$, Julia $J=1/\hbar$; both use
$T=20$, $\Delta t=0.1$ ($n=200$), $\varepsilon=10^{-10}$,
$N\_E\in\{5,10,25,50,100\}$,
`use_symmetric_Trotter` / `sys_alg=Trotter{2}()`, `combine_alg=Trotter{2}()`,
`:zipup_cpp`. System $\rho\_S(0)=\tfrac12(I+\sigma\_x)$. Couplings
$J\_k=J/N$. Polarisation:

| Bath | C++ |
| :--- | :--- |
| fully polarised, all $\lvert\uparrow\rangle\_z$ | `RandomSpin_set_initial_dir 0 0 1` |
| partial, Boltzmann-filtered pure states vs $\mathbf B=(0,0,20)$ | `RandomSpin_B_init 0 0 20`, `RandomSpin_T 11.604522` |
| unpolarised independent pure states | omit both; `RandomSpin_seed 1` |

Partial states are still **pure**, not mixed Gibbs. Templates:
[`cpp/central_spin_polarised.param`](cpp/central_spin_polarised.param),
[`cpp/central_spin_partial.param`](cpp/central_spin_partial.param),
[`cpp/central_spin_unpolarised.param`](cpp/central_spin_unpolarised.param).

### Lorentzian spin-boson

SI Sec. S.4.D independent-boson grid. Production $C/\Omega^2=0.2$ (SI used
$0.1$), $M=5$ Fock levels (fixed in $T$), $\Omega T=8$,
$\Omega\Delta t=0.1$, $\varepsilon=10^{-8}$,
$N\_E\in\{5,10,20,50,100\}$, $k\_B T/\Omega\in\{0,0.5,1,3\}$,
$0\le\omega/\Omega\le 7.5$.

$$
J(\omega)=\frac{C}{\pi}\frac{\gamma}{(\omega-\omega\_c)^2+\gamma^2},\quad
\omega\_c=\Omega,\ \gamma=0.1\,\Omega,
$$
$$
g\_k=\sqrt{J(\omega\_k)\,\Delta\omega},\quad
\Delta\omega=7.5\,\Omega/N\_E.
$$
$$
H\_S=\tfrac{\Omega}{2}\sigma\_x,\quad
\rho\_S(0)=\lvert g\rangle\langle g\rvert,\quad
H\_{B\_k}=\omega\_k a\_k^\dagger a\_k,
$$
$$
H\_{\mathrm{int},k}=g\_k(a\_k+a\_k^\dagger)\lvert e\rangle\langle e\rvert
+\frac{g\_k^2}{\omega\_k}\lvert e\rangle\langle e\rvert.
$$

At $T=0$ every mode is vacuum. C++: [`cpp/spinboson.param`](cpp/spinboson.param)
(`Boson_J_type lorentzian`, `Boson_J_scale 0.2`, `Boson_J_shift 1.0`). Julia
uses the same closed form in [`common.jl`](common.jl).

## Setup

```bash
sudo apt install libeigen3-dev   # headers at /usr/include/eigen3
cd ACE && make && cd ..          # libACE.so, bin/ACE, tools/PTB_analyze
export PATH="$PWD/ACE/bin:$PWD/ACE/tools:$PATH"
ACE                              # usage, not command-not-found
```

Rebuild with `cd ACE && make`. Pin one OpenMP/MKL/OpenBLAS thread and
`julia -t 1`. `ACE_CPUS` / `JULIA_VS_CPP_CPUS` is a core pool (one case per
free core). Completed CSV keys are skipped.

### macOS (Apple Silicon)

Intel MKL has no native arm64 macOS build. The MacBook campaign links ACE
against OpenBLAS with `-DEIGEN_USE_BLAS -DEIGEN_USE_LAPACKE`, so Eigen's
`JacobiSVD` dispatches to `LAPACKE_zgesvd` (the non-MKL half of
`EIGEN_USE_MKL_ALL`). The gitignored `ACE/Makefile` has a Darwin `@rpath` link
and an `OPENBLAS_HOME` block for this.

```bash
brew install gcc openblas
curl -L https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz | tar xz -C ACE/external
cd ACE && make CXX=/opt/homebrew/bin/g++-16 \
  EIGEN_HOME="$PWD/external/eigen-3.4.0" OPENBLAS_HOME=/opt/homebrew/opt/openblas
```

macOS has no core affinity, so cases run unpinned and `ACE_CPUS` only sets how
many run at once (default pool = performance cores). The Julia benchmark
manifest needs `julia +1.12.7`. Write the campaign to its own directory so the
archived `results/{cpp,julia}/central_spin.csv` are never touched:

```bash
export JULIA_VS_CPP_RESULTS=benchmark/JuliaVsC++/results/macbook
ACE_CPUS="0 1 2 3" bash benchmark/JuliaVsC++/run_cpp_central_spin.sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  julia +1.12.7 -t 1 --project=benchmark benchmark/JuliaVsC++/run_julia_central_spin.jl
```

Run C++ first: partial and unpolarised Julia baths read the C++ orientation
dumps from `$JULIA_VS_CPP_RESULTS/cpp/`.

## Run

```bash
bash benchmark/JuliaVsC++/run_cpp_central_spin.sh
bash benchmark/JuliaVsC++/run_cpp_spinboson.sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  julia -t 1 --project=benchmark benchmark/JuliaVsC++/run_julia_central_spin.jl
bash benchmark/JuliaVsC++/run_julia_spinboson_parallel.sh
julia --project=benchmark benchmark/JuliaVsC++/plot_results.jl
```

Subsets and pinning:

```bash
ACE_CPUS="0 1 2 3" NS="5 10" POLS="polarised" \
  bash benchmark/JuliaVsC++/run_cpp_central_spin.sh
ACE_CPUS="0 1 2 3" TEMPS="0.5 1.0" NS="5 10" \
  bash benchmark/JuliaVsC++/run_cpp_spinboson.sh
JULIA_VS_CPP_SMOKE=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  julia -t 1 --project=benchmark benchmark/JuliaVsC++/run_julia_central_spin.jl
ACE_CPUS="16 17 18 19" TEMPS="0 0.5" NS="5 10" \
  bash benchmark/JuliaVsC++/run_julia_spinboson_parallel.sh
```

C++ wall time is `/usr/bin/time` with `dont_propagate`. $D\_{\max}$ is the last
`Maxdim` in the log, or `PTB_analyze -read_PT …`. Julia times
`build_process_tensor` after an untimed warmup (BenchmarkTools `evals=1`);
bath setup / orientation parsing is `setup_s`. Partial and unpolarised Julia
baths import C++ `*_orientations.txt` dumps. Smoke runs write a throwaway CSV
`*smoke*.csv` that is is gitignored. These runs are just to ensure we check the wiring of 
scripts and parameters work well before running the full benchmarking.

Serial spin-boson Julia: [`run_julia_spinboson.jl`](run_julia_spinboson.jl)
with `JULIA_VS_CPP_TEMPS` / `JULIA_VS_CPP_NS`. The parallel wrapper
[`run_julia_spinboson_parallel.sh`](run_julia_spinboson_parallel.sh) writes
one fragment per case and merges after workers finish.

Plotter [`plot_results.jl`](plot_results.jl): filled/dashed = Julia,
hollow/dotted = C++. Spin-boson series are grouped by $T$ and labelled by
$N\_E$. Figures:
[`results/julia_vs_cpp.pdf`](results/julia_vs_cpp.pdf),
[`results/julia_vs_cpp.png`](results/julia_vs_cpp.png).

## Outputs

Each CSV row is one PT construction.

| File | |
| :--- | :--- |
| [`results/cpp/central_spin.csv`](results/cpp/central_spin.csv) | C++ central spin |
| [`results/cpp/spinboson.csv`](results/cpp/spinboson.csv) | C++ spin-boson |
| [`results/julia/central_spin.csv`](results/julia/central_spin.csv) | Julia central spin |
| [`results/julia/spinboson.csv`](results/julia/spinboson.csv) | Julia spin-boson |

| Column | Meaning |
| ---: | :--- |
| `build_s` | PT construction (the fight) |
| `contract_s` | 0 under `dont_propagate` / build-only Julia |
| `io_s` | `PTB_analyze` or other I/O |
| `total_s` | C++ process wall; Julia `setup_s + build_s` |

C++ also writes `.log`, `.time`, `.pt`, and orientation dumps (`.pt` gitignored).
Head-to-head: one BLAS/OpenMP thread, `julia -t 1`, distinct `taskset` CPUs.
Runners print Julia / ProcessTensors / ITensors / `g++` / Eigen at startup.

## Sanity (Table S.3.1)

Not a build-only comparison. See
[`sanity_central_spin/README.md`](sanity_central_spin/README.md) and
[`sanity_spin_boson/README.md`](sanity_spin_boson/README.md)
([`published_runtimes.csv`](sanity_central_spin/published_runtimes.csv),
[spin-boson](sanity_spin_boson/published_runtimes.csv)).

```bash
bash benchmark/JuliaVsC++/run_sanity_cpp_vs_published.sh
bash benchmark/JuliaVsC++/run_sanity_cpp_julia_ed.sh
```

Published CS: $\Delta t=0.01\,\hbar/J$, $t\_{\mathrm{final}}=20\,\hbar/J$.
This production sweep is $\Delta t=0.1$, $n=200$. Published IB:
[ACE Morse param](https://github.com/mcygorek/ACE/blob/master/examples/NatPhys2022/04_morse/01_independent_boson_T0.5_M5_Jscale0.1.param)
($N=101$, $C/\Omega^2=0.1$, $t\_{\mathrm{final}}=10$, $\varepsilon=10^{-7}$).
This sweep is $C/\Omega^2=0.2$, $N\_E\in\{5,10,20,50,100\}$,
$t\_{\mathrm{final}}=8$, $\varepsilon=10^{-8}$.

Beating laptop Table S.3.1 times on a Threadripper only shows a healthy
`-O3` C++ binary, not that Julia is faster.

## Matching caveats

- Fully polarised CS is a like-for-like microscopic match. Unpolarised /
  partial Julia uses the C++ orientation dumps.
- Julia `:zipup_cpp` is C++ `add_modes` (truncate then join) with ITensors
  `gesdd`. C++ uses Eigen JacobiSVD. Do not mix in `:canonzip`.
- `sys_alg=Trotter{2}()` is the system–bath split. C++ mode maps are exact
  per mode then joined. Julia embeds $H\_S$ after the bath sweep (does not
  change $D\_{\max}$).
