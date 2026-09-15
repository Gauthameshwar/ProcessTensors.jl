# Independent-boson / spin-boson sanity

Published Table S.3.1 keys: `dt=0.1`, `C/Ω²=0.1`, `k_BT=0.5Ω`, `M=5`,
`ε=10^{-7}`, Lorentzian on `[0,7.5]`. The C++ runtime case uses `N_E=101`
and `te=10`. The ED accuracy case uses `N_E=2` and `te=2`.

```bash
bash benchmark/JuliaVsC++/run_sanity_cpp_vs_published.sh
bash benchmark/JuliaVsC++/run_sanity_cpp_julia_ed.sh
```

`published/` holds the construction-only ACE input. `published_runtimes.csv`
is Table S.3.1 for the independent-boson / Morse family.
