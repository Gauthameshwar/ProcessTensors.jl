# ACE SVD audit, phase 1

Diagnosis of the N=5 partially polarised central-spin build. This directory
does not modify the production benchmark CSVs or the ACE truncation rules.

```bash
tmux new-window -d -t gesvd-bench -n ace-svd-audit \
  'bash benchmark/JuliaVsC++/profiling/run_all.sh; echo AUDIT_EXIT:$?; exec bash'
```

The measurement is the N=5 partial central-spin build. It does not wait for
the spin-boson gesvd jobs. It pins one idle core whose sibling hyperthread is
also idle, and it refuses cores those jobs already occupy. Julia is
`julia +1.12.7 -t 1` with one BLAS/OpenMP thread.

Outputs:

- `results/reference.csv` — production `build_process_tensor` gesvd timing
- `results/per_svd_gesvd.csv`, `results/per_svd_gesdd.csv` — one row per SVD
- `results/stage_times.csv` — contraction, matricisation, LAPACK, truncation, reassembly
- `fixtures/svd_*.bin` — small, median-cost, and expensive matrices
- `results/replay.csv` — Julia gesvd, Julia gesdd, ITensors gesvd, LP64 LAPACKE gesvd
- `logs/cpp_zgesvd.csv` — C++ `LAPACKE_zgesvd` calls, if the preload run succeeds
- `report.md`, `results/svd_profile.png`

The C++ harness links `~/.local/openblas` and is built by the script. It does
not rebuild OpenBLAS.
