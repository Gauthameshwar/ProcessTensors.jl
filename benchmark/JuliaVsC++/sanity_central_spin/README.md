# Central-spin sanity

Published Table S.3.1 discretisation (`dt=0.01`, `te=20` for the C++
runtime check; `eps=10^{-10}` or `10^{-13}`). Both entry-point scripts live
one directory up and stay single-threaded.

```bash
bash benchmark/JuliaVsC++/run_sanity_cpp_vs_published.sh
bash benchmark/JuliaVsC++/run_sanity_cpp_julia_ed.sh
```

`published/` holds construction-only ACE inputs (`dont_propagate`,
`use_symmetric_Trotter`). `published_runtimes.csv` is Table S.3.1 for
this model. `central_spin_sanity.param` is the N=3 ED accuracy case. The
C++ couplings there use `J_k=ħ/N` so they match Julia `J=1` in
ProcessTensors.jl units.
