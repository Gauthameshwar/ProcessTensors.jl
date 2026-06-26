# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors #src
# SPDX-License-Identifier: MIT #src
# #src
# File: docs/literate/tutorials/08_multitime_correlations.jl #src
# Contributor: Gauthameshwar S. #src
# #src
# Literate tutorial source: two-time correlations (placeholder). #src

# # Multi-Time Correlations
#
# This tutorial will evaluate sequential two-time correlators by contracting a
# process tensor with `two_time_correlation_seq`.

using ProcessTensors

sys = siteinds("S=1/2", 1)
H = OpSum() + (0.5, "Sz", 1)
system = spin_system(sys, H)
pt = build_process_tensor(system; dt=0.05, nsteps=4)
ρ0 = to_dm(MPS(sys, "Up"))

O = OpSum() + (1.0, "Sz", 1)
seq = two_time_correlation_seq(pt, (O, 2), (O, 0); rho0=ρ0)
val = evaluate_process(pt, seq)

@assert val isa ComplexF64
@assert isfinite(real(val))

# ## Planned sections
#
# - Heisenberg-picture correlators and instrument placement
# - `left_action` / `right_action` and `ProductInstrument`
# - Comparison with small exact references
