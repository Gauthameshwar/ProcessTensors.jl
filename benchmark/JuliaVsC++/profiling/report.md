# ACE SVD performance audit, phase 1

Partial central spin, N=5, `dt=0.1`, `t_final=20`, `ε=1e-10`, `:zipup_cpp`, C++ orientations. Julia 1.12.7, one thread. The reference build uses the production `build_process_tensor` path. The stage breakdown uses a profiling copy of the zip-up driver that calls `NDTensors.svd`, which is the factorisation `ITensors.svd` calls. No cutoff, truncation rule, or solver default was changed.

## Provenance

```
julia 1.12.7
ProcessTensors 0.2.0
ITensors 0.9.31
ITensorMPS 0.4.1
blas /home/gautham/.julia/juliaup/julia-1.12.7+0.x64.linux.gnu/bin/../lib/julia/libopenblas64_.so
blas_config LBTConfig([ILP64] libopenblas64_.so)
blas_int_bits 64
julia_threads 1
blas_threads 1
omp 1
openblas_env 1
cpu 159
probe_lib 
hostname tieshan

```

C++ ACE enters LAPACK through Eigen `JacobiSVD` with `ComputeThinU | ComputeThinV`, and this build replaces that with `LAPACKE_zgesvd` at job `S`,`S`. Julia's `LinearAlgebra.QRIteration` also calls `gesvd!` with job `S`,`S`. Julia's BLAS is ILP64. The C++ replay harness is LP64 OpenBLAS from `~/.local/openblas`.

case: partial central spin N=5
pinned_cpu: 159
excluded_cpus: 70 71 72 73 150 151 152 153
spin-boson gesvd jobs were left running on their own cores and were not waited on.
This audit does not share those cores or their sibling hyperthreads.


## Validity

Uninstrumented Julia gesvd build: 159.4 s, `D_max=65`.
Instrumented Julia gesvd build: 153.7 s, `D_max=65`.
Instrumented Julia gesdd build: 128.7 s, `D_max=65`.
The stored production timings for this case were C++ gesvd 53.48 s, Julia gesdd 55.28 s, Julia gesvd 170.69 s, all with `D_max=65`.

## Answers

1. Julia gesvd performed 1990 SVDs and Julia gesdd performed 1990. Their (m, n) sequences are identical. C++ LAPACKE_zgesvd was called 1990 times. The forward 995 shapes match Julia. The backward 995 shapes are the transpose of Julia's, so the unordered pairs {m, n} match. `logs/cpp_zgesvd.csv`.
2. LAPACK is 60.34% of the instrumented Julia gesvd build (92.75 s of 153.7 s).
3. No. On the most expensive matrix the isolated Julia/C++ gesvd ratio is 1.241, so the full-build gap is not reproduced by one LAPACK call. Julia gesvd / Julia gesdd on that matrix is 1.355. The ITensors wrapper / raw Julia gesvd ratio is 0.8258. ACE retained ranks: small julia_gesvd=4, julia_gesdd=4, itensors_gesvd=4, lapacke_gesvd=4; median julia_gesvd=25, julia_gesdd=25, itensors_gesvd=25, lapacke_gesvd=25; expensive julia_gesvd=144, julia_gesdd=144, itensors_gesvd=144, lapacke_gesvd=144. Singular values: median singular-value relative difference 8.628669804912782e-7; expensive singular-value relative difference 1.5503954881714979e-6; small singular-value relative difference 4.000511795427902e-7. Reconstruction residuals are the `rel_recon` column of `results/replay.csv`.
4. In the instrumented gesvd build, index combination and permutation took 16.02 s, `NDTensors.matrix` took 0.008317 s, `NDTensors.svd_catch_error` (the `LinearAlgebra.svd` call, including its workspace query) took 92.75 s, ACE truncation took 8.58 s, and ITensor reassembly took 5.414 s. Joining the mode tensors took 9.131 s. Unassigned build time is 21.8 s.
5. Julia uses ILP64 `libopenblas64_.so` (integer width 64 bits). C++ uses LP64 OpenBLAS 0.3.30. Workspace queries returned the same `lwork` on each replayed matrix: median julia_gesvd lwork=4536; median lapacke_gesvd lwork=4536; expensive julia_gesvd lwork=46000; expensive lapacke_gesvd lwork=46000; small julia_gesvd lwork=280; small lapacke_gesvd lwork=280. Workspace size does not explain a factor of about three. The ABI and the OpenBLAS version are still confounded: this run does not rebuild either library.
6. Julia gesvd and gesdd issue the same number of factorisations on the same shapes, and an isolated gesvd call is not about 3× a C++ LAPACKE call. Workspace sizes match, so `lwork` does not explain a factor of three. The instrumented gesdd wall time is not the stored production gesdd time, so this profiling copy does not reproduce that historical gap. The ABI and the OpenBLAS version remain confounded. The next decisive experiment is to replay the expensive matrix with one OpenBLAS build compiled twice, LP64 and ILP64, and, separately, with 0.3.29 and 0.3.30 at fixed ABI. `perf stat` on `zgesvd_64_` versus `zgesvd_` for that matrix would show whether the extra time is inside the blocked QR iteration.

## Files

- `results/reference.csv`, `results/per_svd_gesvd.csv`, `results/per_svd_gesdd.csv`, `results/stage_times.csv`
- `results/replay.csv`, `results/svd_profile.png`
- `fixtures/svd_*.bin`, `logs/`

Figure: `results/svd_profile.png`.
