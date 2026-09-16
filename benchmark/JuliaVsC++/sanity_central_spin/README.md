# Central-spin sanity

Published Table S.3.1 discretisation (`dt=0.01`, `te=20` for the C++
runtime check; `eps=10^{-10}` or `10^{-13}`). Both entry-point scripts live
one directory up and stay single-threaded.

```bash
bash benchmark/JuliaVsC++/run_sanity_cpp_vs_published.sh
bash benchmark/JuliaVsC++/run_sanity_cpp_julia_ed.sh
```

- [`../run_sanity_cpp_vs_published.sh`](../run_sanity_cpp_vs_published.sh)
- [`../run_sanity_cpp_julia_ed.sh`](../run_sanity_cpp_julia_ed.sh)
- [`run_central_spin_sanity.jl`](run_central_spin_sanity.jl)
- [`published/`](published/) holds construction-only ACE inputs (`dont_propagate`,
  `use_symmetric_Trotter`)
- [`published_runtimes.csv`](published_runtimes.csv) is Table S.3.1 for this model
- [`central_spin_sanity.param`](central_spin_sanity.param) is the N=3 ED accuracy case

The C++ couplings there use `J_k=ħ/N` so they match Julia `J=1` in
ProcessTensors.jl units.

Last ED check ([`results/summary.txt`](results/summary.txt)): $N=3$,
`:zipup_cpp`, max rel. Frobenius vs ED $\approx 8.6\times 10^{-8}$,
Julia vs C++ $\approx 3.5\times 10^{-10}$.
