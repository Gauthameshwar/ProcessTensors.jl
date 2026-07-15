# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: scripts/terminal/spinner_demo.jl
# Contributor: Gauthameshwar S.
#
# Demonstrates the package header-spinner + child-bar layout used by
# build_process_tensor: one spinner stays on the top line until the workflow
# ends, while a determinate bar updates underneath during assembly.
#
# Run with:
# julia --project=. -t auto scripts/terminal/spinner_demo.jl

using ProcessTensors

const STAGE_SECONDS = 2.0

@info "Starting multi-stage spinner demo"
@info "The top spinner should keep rotating while the assembly bar runs below it."

reporter = ProcessTensors._progress_reporter(true)
started = time()

try
    ProcessTensors._ensure_spinner!(
        reporter,
        "Building process tensor — starting",
    )
    sleep(STAGE_SECONDS * 2)

    ProcessTensors._with_progress(
        reporter,
        "Building process tensor — assembling cores",
        20,
    ) do update!
        for step in 1:20
            update!(step)
            sleep(0.12)
        end
    end

    ProcessTensors._ensure_spinner!(
        reporter,
        "Building process tensor — closing bath boundaries",
    )
    sleep(STAGE_SECONDS)
finally
    ProcessTensors._clear_stage!(reporter)
end

elapsed = time() - started
@info "Spinner demo finished" elapsed_seconds=elapsed
println("Multi-stage spinner demo held for ", round(elapsed; digits=2), " s.")
