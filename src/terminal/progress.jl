# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: src/terminal/progress.jl
# Contributor: Gauthameshwar S.
#
# Aggregates the terminal progress backend in dependency order and exposes the
# five contributor-facing reporting macros to the parent module.

include("types.jl")
include("tty_renderer.jl")
include("progressmeter_backend.jl")
include("reporting.jl")
include("macros.jl")
