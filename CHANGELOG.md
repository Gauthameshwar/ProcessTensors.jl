# Changelog

All notable changes to [ProcessTensors.jl](https://github.com/Gauthameshwar/ProcessTensors.jl) are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — v0.2.0

### Breaking

* **`sys_alg` is the PT timestep sandwich order**, not a TEBD
  factorization of the free-system map. Single-site system maps in dense PT
  cores are always Exact ED. Default `sys_alg=Trotter{1}()` keeps the asymmetric
  bake-in ``Q·M(Δt)``; `Trotter{2}()` uses ``M(Δt/2)·Q·M(Δt/2)``. Prefer
  `sys_alg=Trotter{2}()` when time-discretization error dominates.
* **Always-embedded system propagation; `SystemPropagation` → `UnitaryPropagation`.**
  Process-tensor slabs always include the system's one-step Liouville map.
  The `embed_system_propagation` keyword is removed.
  Schedule-side unitary control uses the new public instrument `UnitaryPropagation`
  (`unitary_propagation`) instead of `SystemPropagation` / `system_propagation`.
  `default_schedule` inserts `IdentityOperation()` between steps (no extra system
  propagator). For a bath-only / identity system map, construct the system with an
  empty `OpSum()` Hamiltonian; the single-mode process-tensor tutorial documents
  the embedded-propagation and identity-schedule conventions.

### Added

* **Terminal progress and verbose operational logging.** `build_process_tensor`,
  `create_instruments`, `evaluate_process`, `evolve`, and `tebd` accept
  `progress=:auto|true|false` for transient ProgressMeter feedback and
  `verbose=true` for persistent structured Julia logs. `:auto` enables meters
  only on interactive non-CI terminals; all stages clear on completion or error.
  Small interactive demonstrations live under `scripts/terminal/`.
* **`OpenInput` / `OpenInOut` bookkeeping instruments.** Like `OpenOutput`, they
  materialize as `ITensor(1.0)` and leave declared process-tensor legs
  uncontracted. Aliases: `open_input`, `open_inout`.
* **`open_leg_info(pt, seq)`** reports claimed, missing, and open legs with
  dimensions before contraction.
* **`instrument_leg_maps(pt, seq)`** thin overload of the seq-first canonical API.

### Changed

* **`instrument_leg_maps`** now uses leg-coverage dispatch from schedule slots
  (no `ProcessTensor` required for the canonical method).
* **`create_instruments`** rewrites the schedule (pairing / terminal
  `IdentityOperation` → `OpenOutput`) then materializes every slot only through
  `instrument_itensor`, including a `SingleLegInstrument` evolve-slot method that
  selects the matching PT leg from `leg_plev`.
* **`evaluate_process`** returns `ComplexF64` / `MPO{Liouville}` / `ITensor`
  according to the number of uncontracted system legs (0 / 1 / ≥2). Optional
  `verbose=true` logs expected open legs.

### Fixed

* **`OpenOutput` is bookkeeping only.** It no longer materializes as a
  `TraceOut` on the next input leg. `instrument_itensor(::OpenOutput)` returns
  the scalar no-op `ITensor(1.0)`, `create_instruments` includes the terminal
  schedule slot (`length = nsteps + 1`), and `evaluate_process` contracts the
  full instrument chain so any open output index remains naturally after the
  loop (no intermediate cut / early break).

### Changed

* Reorganized process-tensor source files into `src/process_tensor/` and dense
  core construction into `src/builders/`, with `build_process_tensor` dispatching
  through `method=Dense()` by default.
* Split instruments into lazy schedule definitions and ITensor materialization
  files under the existing `Instruments` submodule at `src/instruments/`.
  `create_instruments` now lives with ITensor instrument materialization while
  keeping the same public name and behavior.
* Multi-site `build_process_tensor` currently errors: always-embedded construction
  requires a single-site system until a multi-site path is added.

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
