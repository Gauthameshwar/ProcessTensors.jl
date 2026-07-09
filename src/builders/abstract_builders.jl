# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: src/builders/abstract_builders.jl
# Contributor: Gauthameshwar S.
#
# Defines process-tensor builder interfaces and the dense builder selector.

"""
    AbstractPTBuilder

Abstract selector for process-tensor construction backends used by
[`build_process_tensor`](@ref).
"""
abstract type AbstractPTBuilder end

"""
    Dense()

Dense joint-Liouville process-tensor builder for no-bath, single-mode, and
small multimode environments.
"""
struct Dense <: AbstractPTBuilder end
