# Independent-boson / spin-boson sanity

Published Table S.3.1 keys: `dt=0.1`, `C/Ω²=0.1`, `k_BT=0.5Ω`, `M=5`,
`ε=10^{-7}`, Lorentzian on `[0,7.5]`. The C++ runtime case uses `N_E=101`
and `te=10`. The ED accuracy case uses `N_E=2` and `te=2`.

```bash
bash benchmark/JuliaVsC++/run_sanity_cpp_vs_published.sh
bash benchmark/JuliaVsC++/run_sanity_cpp_julia_ed.sh
```

- [`../run_sanity_cpp_vs_published.sh`](../run_sanity_cpp_vs_published.sh)
- [`../run_sanity_cpp_julia_ed.sh`](../run_sanity_cpp_julia_ed.sh)
- [`run_spinboson_sanity.jl`](run_spinboson_sanity.jl)
- [`spinboson_sanity.param`](spinboson_sanity.param)
- [`published/`](published/) holds the construction-only ACE input
- [`published_runtimes.csv`](published_runtimes.csv) is Table S.3.1 for the
  independent-boson / Morse family

Last ED check ([`results/summary.txt`](results/summary.txt)): $N\_E=2$,
$M=5$, max rel. Frobenius vs ED $\approx 2.1\times 10^{-5}$,
Julia vs C++ $\approx 6.5\times 10^{-7}$.
