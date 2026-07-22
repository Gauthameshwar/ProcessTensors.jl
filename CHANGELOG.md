# Changelog

All notable changes to [ProcessTensors.jl](https://github.com/Gauthameshwar/ProcessTensors.jl) are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — v0.2.0

### Breaking

* **Renamed Liouvillian constructors (compat aliases retained).** Prefer
  `liouvillian_opsum`, `liouvillian_mpo`, and `liouvillian_propagator` over the
  previous public names `OpSum_Liouville`, `MPO_Liouville`, and
  `liouvillian_propagator_itensor`. The old names remain exported as deprecated
  forwarding aliases for one migration window and will be removed in a later
  `0.y` release. Numerical behaviour is unchanged.
* **`AbstractSpace` is no longer root-exported.** It remains available as the
  qualified developer interface `ProcessTensors.AbstractSpace`.
  `Hilbert`, `Liouville`, `MPS`, `MPO`, `AbstractMPS`, and `AbstractMPO` stay
  root-exported.
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
  `default_schedule` inserts `identity_operation()` between steps (no extra system
  propagator). For a bath-only / identity system map, construct the system with an
  empty `OpSum()` Hamiltonian; the single-mode process-tensor tutorial documents
  the embedded-propagation and identity-schedule conventions.
* **Canonical constructors vs lowercase user-facing constructors.** Domain types such as
  `BosonicMode`, `SpinBath`, `TraceOut`, and `UnitaryPropagation` keep one
  strict direct-construction shape. Relaxed defaults, keyword forms, and
  inferred arguments live only on the lowercase user-facing constructors (`bosonic_mode`,
  `spin_bath`, `trace_out`, `unitary_propagation`, …). Zero-argument helpers
  such as `TraceOut()` and `IdentityOperation()` are removed; use
  `trace_out()` and `identity_operation()` instead. Both types and user-facing constructors
  remain exported.
* **Process-tensor / instrument API boundaries.** `all_pt_legs_contracted` is
  renamed to `isfullycontracted`. `generate_pt_legs` is internalised as
  `_generate_pt_legs`. `AbstractPTBuilder` is no longer root-exported (use
  `ProcessTensors.AbstractPTBuilder`); `Dense` / `Dense()` stay root-exported and
  unchanged. `instrument_leg_maps`, `resolve_instrument`, `instrument_itensor`,
  and `create_instruments` remain public only through
  `ProcessTensors.Instruments` (not root-exported).
  One-open-leg `evaluate_process` now returns `MPS{Liouville}` instead of
  `MPO{Liouville}`.
* **`propagator_itensor_from_gates` is contraction-only.** The `:basis` /
  `:auto` materialisation paths and the `materialize_method` keyword on
  `liouvillian_propagator` are removed. Trotter gate lists are always contracted
  with index promotion.
* **Time-evolution root API is `tebd` / `tdvp` only.** `Exact` and `Trotter`
  are no longer root-exported; import them from `ITensors.Ops` when selecting an
  algorithm. `trotter_gates` and `propagator_itensor_from_gates` remain defined
  for qualified advanced use (`ProcessTensors.trotter_gates`, …) but are not
  exported. Unused TDVP utilities from ITensorMPS
  (`promote_itensor_eltype`, `convert_leaf_eltype`, `argsdict`, `sim!`) are no
  longer imported in `tdvp.jl` or root-exported.

### Added

* **Laser-driven midpoint TDVP example.** A callable `OpSum` represents a
  Gaussian time-dependent drive, while a dedicated script generates
  energy-density and excitation-density plots. Redundant driven two-level and
  kicked-Ising stub cards were removed from the docs sidebar.
* **`OpenInput` / `OpenInOut` bookkeeping instruments.** Like `OpenOutput`, they
  materialize as `ITensor(1.0)` and leave declared process-tensor legs
  uncontracted. Aliases: `open_input`, `open_inout`.
* **`open_leg_info(pt, seq)`** reports claimed, missing, and open legs with
  dimensions before contraction.
* **`instrument_leg_maps(pt, seq)`** thin overload of the seq-first canonical API.

### Changed

* Internalised Liouvillian OpSum builders as `_build_liouvillian_opsum` and
  `_build_liouvillian_opsum_from_lindblad` (unexported implementation details).
* **`instrument_leg_maps`** now uses leg-coverage dispatch from schedule slots
  (no `ProcessTensor` required for the canonical method).
* **`create_instruments`** rewrites the schedule (pairing / terminal
  `identity_operation` → `open_output`) then materializes every slot only through
  `instrument_itensor`, including a `SingleLegInstrument` evolve-slot method that
  selects the matching PT leg from `leg_plev`.
* **`evaluate_process`** returns `ComplexF64` / `MPS{Liouville}` / `ITensor`
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
