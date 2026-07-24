# Changelog

All notable changes to [ProcessTensors.jl](https://github.com/Gauthameshwar/ProcessTensors.jl) are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — v0.2.0

### Breaking

* **Renamed Liouvillian constructors.** Prefer `liouvillian_*`; old names remain deprecated aliases for one migration window.
* **`AbstractSpace` is no longer root-exported.** Use `ProcessTensors.AbstractSpace`.
* **`sys_alg` is the PT timestep sandwich order**, not a TEBD factorization of the free-system map.
* **Always-embedded system propagation.** `SystemPropagation` → `UnitaryPropagation`; `embed_system_propagation` is removed.
* **Canonical vs lowercase constructors.** Relaxed forms live only on lowercase constructors; zero-argument type helpers are removed.
* **Process-tensor / instrument API boundaries.** `isfullycontracted` rename; materialisation stays under `Instruments`.
* **`propagator_itensor_from_gates` is contraction-only.** Materialisation keywords are removed.
* **Time-evolution root API is `tebd` / `tdvp` only.** `Exact`, `Trotter`, and `trotter_gates` are not root-exported.
* **Compact tensor-network root surface.** Advanced TN surgery is not provided; use ITensorMPS on cores when needed.

### Added

* **Laser-driven midpoint TDVP example** with a callable `OpSum` drive and density plots.
* **`OpenInput` / `OpenInOut`** bookkeeping instruments (aliases: `open_input`, `open_inout`).
* **`open_leg_info(pt, seq)`** reports claimed, missing, and open legs before contraction.
* **`instrument_leg_maps(pt, seq)`** thin overload of the seq-first canonical API.

### Changed

* Internalised Liouvillian OpSum builders as unexported `_build_liouvillian_opsum*` helpers.
* **`instrument_leg_maps`** dispatches from schedule slots without requiring a `ProcessTensor`.
* **`create_instruments`** rewrites the schedule, then materializes every slot through `instrument_itensor`.
* **`evaluate_process`** returns `ComplexF64` / `MPS{Liouville}` / `ITensor` for 0 / 1 / ≥2 open legs.
* Reorganized process-tensor sources under `src/process_tensor/` and dense builders under `src/builders/`.
* Split instruments into lazy schedule definitions and ITensor materialization under `Instruments`.
* Multi-site `build_process_tensor` currently errors until a multi-site path is added.

### Fixed

* **`OpenOutput` is bookkeeping only.** It materializes as `ITensor(1.0)` and no longer acts as a `TraceOut`.
* **Bugfix:** `evaluate_process` now returns `MPS{Liouville}` (not `MPO{Liouville}`) when called with one open leg.

## v0.1.0 - 2026-07-03

### First Public Preview of MPS-Based Process Tensors in Julia

This is the first public development release of `ProcessTensors.jl`.

`ProcessTensors.jl` provides MPS/MPO-based tools for open quantum dynamics, Liouville-space simulation, and process-tensor workflows in Julia. The package is built on top of `ITensors.jl` and `ITensorMPS.jl`, adding Hilbert/Liouville-space wrappers, system and bath abstractions, instrument sequences, process-tensor construction, reduced evolution, and multi-time correlation utilities.

This is a `0.1.0` release, so the API should be considered experimental and may change in future minor releases.

### Highlights

* Added `MPS{Hilbert}`, `MPS{Liouville}`, `MPO{Hilbert}`, and `MPO{Liouville}` wrappers around ITensorMPS objects.
* Added Hilbert-to-Liouville conversion tools, including `liouv_sites`, `to_dm`, `to_liouville`, and `to_hilbert`.
* Added Liouville-space operator construction through `OpSum_Liouville` and `MPO_Liouville`.
* Added spin and bosonic system abstractions.
* Added spin and bosonic bath-mode abstractions.
* Added dense single-mode and small multimode process-tensor construction.
* Added instrument-sequence support, including state preparation, observable measurement, trace-out, identity operations, open outputs, and system propagation.
* Added process-tensor contraction through `evaluate_process`.
* Added reduced-system evolution through `evolve`.
* Added sequential multi-time correlation utilities through `two_time_correlation_seq`.
* Added TEBD/TDVP forwarding and time-evolution support for Hilbert- and Liouville-space workflows.
* Added tutorials, theory pages, API documentation, and runnable example scripts.

### Documentation

The documentation includes:

* introductory theory pages for tensor networks, Liouville space, and process tensors,
* tutorials for ITensor basics, MPS/MPO basics, Liouville-space workflows, unitary dynamics, dissipative dynamics, and process tensors,
* example pages for TEBD/TDVP dynamics, dissipative systems, spin-bath process tensors, and multi-time correlations,
* an API reference for the public ProcessTensors interface.

### Testing and validation

This release includes tests for:

* MPS/MPO wrappers and constructors,
* network operations and orthogonality utilities,
* Hilbert/Liouville conversion,
* Liouvillian construction,
* TEBD/TDVP workflows,
* systems, baths, and instruments,
* process-tensor structure and contraction,
* causality checks,
* dense exact-diagonalization comparisons,
* multimode spin-bath examples,
* multi-time correlations,
* selected comparisons against QuantumOptics.jl.

### Known limitations

* This is an initial development release; the public API is not yet stable.
* Process-tensor construction currently targets single-mode and small multimode dense workflows.
* Large-scale non-Markovian environment algorithms such as ACE/PT-TEMPO are planned future extensions.
* Some documentation and API polishing will continue after this release.
* Performance tuning and larger benchmark studies are still ongoing.

### Upgrade notes

This is the first tagged release, so there are no previous versions to migrate from.

Users should expect API refinements in future `0.x` releases as the package grows.
