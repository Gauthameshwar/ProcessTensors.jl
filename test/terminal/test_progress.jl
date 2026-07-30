# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: test/terminal/test_progress.jl
# Contributor: Gauthameshwar S.
#
# Tests the terminal progress backend: strict no-op guarantees, spinner and bar
# lifecycles, cleanup after success and failure, verbose combinations, macro
# hygiene, and dependency isolation of terminal mechanics.
#
# Run with:
# julia --project=. test/runtests.jl

using ProcessTensors
using ITensors
using Logging
using Test

# Dummy tracked workflow used to exercise the full five-macro integration
# pattern (also validates that a new builder can add reporting with macros only).
function _dummy_tracked_workflow(;
    nsteps::Int=3,
    progress::Union{Bool,Symbol},
    verbose::Bool,
    output::IO=IOBuffer(),
)
    run = ProcessTensors._run_reporter(progress, verbose; output=output)
    ProcessTensors._progress_start!(run, "Running dummy workflow"; nsteps=nsteps)
    try
        ProcessTensors.@progress_stage run "Preparing dummy workspace"
        total = 0
        ProcessTensors.@progress_bar run "Advancing dummy trajectory" nsteps begin
            for step in 1:nsteps
                total += step
                ProcessTensors.@progress_update run step (t=step,)
            end
        end
        ProcessTensors.@progress_stage run "Finished dummy workflow" (total=total,)
        return total
    finally
        ProcessTensors.@progress_finish run
    end
end

@testset "terminal progress backend" begin
    @testset "reporter construction" begin
        @test ProcessTensors._run_reporter(false, false) isa ProcessTensors._NoRunReporter
        @test ProcessTensors._run_reporter(false, true).verbose
        @test ProcessTensors._run_reporter(true, false; output=IOBuffer()) isa
              ProcessTensors._TTYRunReporter
        # :auto never activates on a non-TTY output.
        @test ProcessTensors._run_reporter(:auto, false; output=IOBuffer()) isa
              ProcessTensors._NoRunReporter
        @test_throws ArgumentError ProcessTensors._run_reporter(:unsupported, false)
    end

    @testset "no-op guarantees when progress is disabled" begin
        output = IOBuffer()
        run = ProcessTensors._run_reporter(false, false; output=output)
        ProcessTensors._progress_start!(run, "Silent operation"; nsteps=2)
        ProcessTensors.@progress_stage run "Silent stage"
        executed = 0
        ProcessTensors.@progress_bar run "Silent bar" 3 begin
            for step in 1:3
                executed += 1
                ProcessTensors.@progress_update run step (t=step,)
            end
        end
        ProcessTensors.@progress_finish run
        @test executed == 3
        @test isempty(String(take!(output)))
    end

    @testset "spinner lifecycle" begin
        output = IOBuffer()
        run = ProcessTensors._run_reporter(true, false; output=output)
        @test run.state == ProcessTensors._ProgressIdle

        ProcessTensors._progress_start!(run, "Demo operation")
        @test run.state == ProcessTensors._ProgressSpinner
        @test run.task !== nothing
        # The first frame is painted synchronously before the worker starts.
        @test occursin("Demo operation", String(take!(copy(output))))

        ProcessTensors.@progress_stage run "Working stage"
        @test run.description == "Working stage"

        sleep(3 * ProcessTensors._SPINNER_INTERVAL + 0.1)
        @test run.frame > 0

        ProcessTensors.@progress_finish run
        @test run.state == ProcessTensors._ProgressClosed
        @test run.task === nothing
        final_output = String(take!(output))
        @test occursin("Working stage", final_output)
        # Final cleanup leaves the header line erased with the cursor at column 0.
        @test endswith(final_output, "\r\e[2K")

        # Repeated finish is safe and stays closed.
        ProcessTensors.@progress_finish run
        @test run.state == ProcessTensors._ProgressClosed
    end

    @testset "thread selection for the spinner worker" begin
        output = IOBuffer()
        run = ProcessTensors._run_reporter(true, false; output=output)
        ProcessTensors._progress_start!(run, "Thread policy")
        task = run.task
        @test task isa Task
        @test !istaskdone(task)
        n_default = Threads.nthreads(:default)
        n_interactive = Threads.nthreads(:interactive)
        if n_default + n_interactive <= 1
            # Cooperative fallback on the only Julia thread.
            @test task.sticky
        elseif n_interactive > 0
            # Default-pool background worker (main lives in the interactive pool).
            @test !istaskdone(task)
        else
            # Sticky pin onto a non-main default thread.
            @test task.sticky
        end
        ProcessTensors.@progress_finish run
    end

    @testset "bar value formatting is fixed-width" begin
        @test ProcessTensors._format_bar_value(1.1) == "1.10"
        @test ProcessTensors._format_bar_value(1.15) == "1.15"
        @test ProcessTensors._format_bar_value(1.0 + 2.345im) == "1.00+2.35im"
        @test ProcessTensors._format_bar_value(1.0 - 2.345im) == "1.00-2.35im"
        @test ProcessTensors._format_bar_value(128) == "128"
    end

    @testset "raw write branches and cursor helpers" begin
        # IOStream path: POSIX write through a temporary file descriptor.
        mktemp() do path, io
            ProcessTensors._raw_write!(io, "iostream-bytes")
            flush(io)
            @test read(path, String) == "iostream-bytes"

            ProcessTensors._show_cursor!(io)
            flush(io)
            @test endswith(read(path, String), "\e[?25h")
        end

        # stdout / non-IOStream branches: empty writes exercise the fd selection
        # without polluting captured test logs.
        ProcessTensors._raw_write!(stdout, UInt8[])
        ProcessTensors._raw_write!(stderr, UInt8[])

        # Non-TTY endpoints refuse to hide the cursor.
        @test ProcessTensors._hide_cursor!(IOBuffer()) == false

        # Real TTY endpoints hide and restore the cursor. Prefer a PTY slave so
        # headless CI still covers the Base.TTY branch when /dev/tty is absent.
        tty_covered = false
        master = ccall(:posix_openpt, Cint, (Cint,), 2) # O_RDWR
        if master >= 0
            try
                if ccall(:grantpt, Cint, (Cint,), master) == 0 &&
                   ccall(:unlockpt, Cint, (Cint,), master) == 0
                    name_ptr = ccall(:ptsname, Cstring, (Cint,), master)
                    if name_ptr != C_NULL
                        slave = ccall(:open, Cint, (Cstring, Cint), name_ptr, 2)
                        if slave >= 0
                            tty = Base.TTY(RawFD(slave))
                            try
                                @test tty isa Base.TTY
                                # `_raw_write!` routes non-stdout TTY endpoints to
                                # fd 2; capture stderr so ANSI codes stay out of
                                # the test runner log.
                                mktemp() do _, sink
                                    redirect_stderr(sink) do
                                        @test ProcessTensors._hide_cursor!(tty) == true
                                        ProcessTensors._show_cursor!(tty)
                                    end
                                end
                                tty_covered = true
                            finally
                                close(tty)
                            end
                        end
                    end
                end
            finally
                ccall(:close, Cint, (Cint,), master)
            end
        end
        if !tty_covered && ispath("/dev/tty")
            try
                open("/dev/tty"; write=true) do tty
                    if tty isa Base.TTY
                        @test ProcessTensors._hide_cursor!(tty) == true
                        ProcessTensors._show_cursor!(tty)
                        tty_covered = true
                    end
                end
            catch err
                @info "Skipping TTY cursor hide/show coverage" exception = err
            end
        end
        @test tty_covered
    end

    @testset "sticky thread spawn helper" begin
        done = Threads.Atomic{Bool}(false)
        tid = Threads.nthreads() >= 2 ? 2 : 1
        task = ProcessTensors._spawn_sticky_on_thread!(
            () -> (Threads.atomic_xchg!(done, true); nothing),
            tid,
        )
        wait(task)
        @test task.sticky
        @test done[]
        @test istaskdone(task)
    end

    @testset "bar lifecycle" begin
        output = IOBuffer()
        run = ProcessTensors._run_reporter(true, false; output=output)

        # A bar may begin only from the spinner state.
        ProcessTensors._begin_progress_bar!(run, "Too early", 3)
        @test run.state == ProcessTensors._ProgressIdle
        @test run.bar isa ProcessTensors._NoBar

        ProcessTensors._progress_start!(run, "Bar demo")
        result = ProcessTensors.@progress_bar run "Assembling dummy cores" 5 begin
            for step in 1:5
                ProcessTensors.@progress_update run step (t=step / 10,)
                sleep(0.02)
            end
            @test run.state == ProcessTensors._ProgressSpinnerBar
            @test run.bar isa ProcessTensors._ProgressMeterBar
            :done
        end
        @test result === :done
        # Bar cleanup returns to spinner-only display; the spinner survives.
        @test run.state == ProcessTensors._ProgressSpinner
        @test run.bar isa ProcessTensors._NoBar

        captured = String(take!(copy(output)))
        @test occursin("Assembling dummy cores", captured)
        @test occursin("100%", captured)
        # Tracked values stay compact, fixed-width, and inline on the bar line.
        @test occursin("t=0.50", captured)

        # An exception inside the bar body still clears the bar.
        @test_throws ErrorException ProcessTensors.@progress_bar run "Failing bar" 2 begin
            ProcessTensors.@progress_update run 1
            error("expected bar failure")
        end
        @test run.bar isa ProcessTensors._NoBar
        @test run.state == ProcessTensors._ProgressSpinner

        ProcessTensors.@progress_finish run
        @test run.state == ProcessTensors._ProgressClosed
    end

    @testset "bar repaint after display interrupt" begin
        output = IOBuffer()
        bar = ProcessTensors._create_bar(output, "Repaint demo", 4)
        ProcessTensors._update_bar!(bar, 2; values=(:χ => 8,))
        ProcessTensors._repaint_bar!(bar)
        @test ProcessTensors._bar_position(bar) == 2
        @test occursin("Repaint demo", String(take!(copy(output))))

        # Verbose stage during an active bar clears, logs, then repaints.
        run = ProcessTensors._run_reporter(true, true; output=output)
        ProcessTensors._progress_start!(run, "Repaint workflow")
        ProcessTensors._begin_progress_bar!(run, "Active bar", 3)
        ProcessTensors._progress_update!(run, 1)
        @test_logs (:info, r"Checkpoint during bar") match_mode = :any begin
            ProcessTensors.@progress_stage run "Checkpoint during bar"
        end
        @test run.state == ProcessTensors._ProgressSpinnerBar
        ProcessTensors.@progress_finish run
    end

    @testset "full cleanup on success and failure" begin
        output = IOBuffer()
        total = _dummy_tracked_workflow(progress=true, verbose=false, output=output)
        @test total == 6
        final_output = String(take!(output))
        @test !isempty(final_output)
        @test endswith(final_output, "\r\e[2K")

        failing_output = IOBuffer()
        run = ProcessTensors._run_reporter(true, false; output=failing_output)
        err = try
            ProcessTensors._progress_start!(run, "Failing operation")
            try
                ProcessTensors.@progress_bar run "Failing stage" 2 begin
                    ProcessTensors.@progress_update run 1
                    error("expected workflow failure")
                end
            finally
                ProcessTensors.@progress_finish run
            end
            nothing
        catch caught
            caught
        end
        @test err isa ErrorException
        @test occursin("expected workflow failure", sprint(showerror, err))
        @test run.state == ProcessTensors._ProgressClosed
        @test run.bar isa ProcessTensors._NoBar
        @test endswith(String(take!(failing_output)), "\r\e[2K")
    end

    @testset "warnings stay visible while progress is active" begin
        output = IOBuffer()
        run = ProcessTensors._run_reporter(true, false; output=output)
        ProcessTensors._progress_start!(run, "Warning demo")
        @test_logs (:warn, r"visible warning") @warn "visible warning"
        ProcessTensors.@progress_finish run
    end

    @testset "verbose and progress combinations" begin
        # progress=false, verbose=false: silence on both channels.
        quiet_output = IOBuffer()
        @test_logs min_level = Logging.Info _dummy_tracked_workflow(
            progress=false, verbose=false, output=quiet_output,
        )
        @test isempty(String(take!(quiet_output)))

        # progress=false, verbose=true: durable checkpoints, no transient bytes.
        verbose_output = IOBuffer()
        @test_logs (:info, r"Running dummy workflow") (:info, r"Preparing dummy workspace") (
            :info,
            r"Advancing dummy trajectory",
        ) (:info, r"Finished dummy workflow") _dummy_tracked_workflow(
            progress=false, verbose=true, output=verbose_output,
        )
        @test isempty(String(take!(verbose_output)))

        # progress=true, verbose=false: transient bytes, no logs.
        live_output = IOBuffer()
        @test_logs min_level = Logging.Info _dummy_tracked_workflow(
            progress=true, verbose=false, output=live_output,
        )
        @test !isempty(String(take!(live_output)))

        # progress=true, verbose=true: both channels active.
        both_output = IOBuffer()
        @test_logs (:info, r"Running dummy workflow") match_mode = :any _dummy_tracked_workflow(
            progress=true, verbose=true, output=both_output,
        )
        @test !isempty(String(take!(both_output)))
    end

    @testset "macro hygiene and single evaluation" begin
        run_evals = Ref(0)
        desc_evals = Ref(0)
        total_evals = Ref(0)
        get_run() = (run_evals[] += 1; ProcessTensors._NO_RUN_REPORTER)
        get_desc() = (desc_evals[] += 1; "Counted stage")
        get_total() = (total_evals[] += 1; 2)

        value = ProcessTensors.@progress_bar get_run() get_desc() get_total() begin
            ProcessTensors.@progress_update ProcessTensors._NO_RUN_REPORTER 1
            41 + 1
        end
        @test value == 42
        @test run_evals[] == 1
        @test desc_evals[] == 1
        @test total_evals[] == 1

        stage_meta_evals = Ref(0)
        get_meta() = (stage_meta_evals[] += 1; (n=1,))
        ProcessTensors.@progress_stage ProcessTensors._NO_RUN_REPORTER "Stage" get_meta()
        @test stage_meta_evals[] == 1

        # Local variables inside the bar body remain visible afterwards.
        accumulated = 0
        ProcessTensors.@progress_bar ProcessTensors._NO_RUN_REPORTER "Locals" 2 begin
            for step in 1:2
                accumulated += step
            end
        end
        @test accumulated == 3
    end

    @testset "workflow verbose logs are durable and opt-in" begin
        sites = siteinds("S=1/2", 1)
        system = spin_system(sites, OpSum() + (0.2, "Sz", 1))
        @test_logs (:info, r"Built process tensor") match_mode = :any build_process_tensor(
            system;
            dt=0.1,
            nsteps=2,
            progress=false,
            verbose=true,
        )

        pt = build_process_tensor(system; dt=0.1, nsteps=3, progress=false)
        rho0 = to_dm(MPS(sites, ["Up"]))
        @test_logs (:info, r"Evolved reduced system") match_mode = :any evolve(
            pt, rho0; progress=false, verbose=true,
        )
        # Fully silent runs stay silent.
        @test_logs min_level = Logging.Info evolve(pt, rho0; progress=false, verbose=false)
    end

    @testset "dependency isolation of terminal mechanics" begin
        src_root = normpath(joinpath(@__DIR__, "..", "..", "src"))
        terminal_dir = joinpath(src_root, "terminal")
        forbidden = ("ProgressMeter.", "\e[", "/dev/tty", "python3",
                     "BLAS.set_num_threads", "numprintedvalues")
        offenders = String[]
        for (root, _, files) in walkdir(src_root)
            startswith(root, terminal_dir) && continue
            for file in files
                endswith(file, ".jl") || continue
                text = read(joinpath(root, file), String)
                for pattern in forbidden
                    occursin(pattern, text) &&
                        push!(offenders, string(joinpath(root, file), " contains ", repr(pattern)))
                end
            end
        end
        @test isempty(offenders)
    end
end
