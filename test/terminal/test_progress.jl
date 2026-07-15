# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: test/terminal/test_progress.jl
# Contributor: Gauthameshwar S.
#
# Tests terminal progress enablement, refresh behavior, and cleanup after
# successful and failing progress scopes.
#
# Run with:
# julia --project=. test/runtests.jl

using ProcessTensors
using ITensors
using Test

@testset "terminal progress reporter" begin
    @testset "enablement" begin
        quiet_output = IOBuffer()
        quiet = ProcessTensors._progress_reporter(false; output=quiet_output)
        @test !quiet.enabled

        forced_output = IOBuffer()
        forced = ProcessTensors._progress_reporter(true; output=forced_output)
        @test forced.enabled

        auto = ProcessTensors._progress_reporter(:auto; output=IOBuffer())
        @test !auto.enabled
        @test_throws ArgumentError ProcessTensors._progress_reporter(:unsupported)
    end

    @testset "determinate refresh and cleanup" begin
        output = IOBuffer()
        reporter = ProcessTensors._progress_reporter(true; output=output)
        snapshots = String[]
        println(output, "[ Info: before progress")
        ProcessTensors._with_progress(reporter, "Testing bar", 20) do update!
            for step in 1:20
                update!(step)
                sleep(0.01)
                push!(snapshots, String(take!(copy(output))))
            end
        end
        println(output, "[ Info: after progress")
        final_output = String(take!(output))
        @test reporter.bar === nothing
        @test !reporter.spinner_active
        @test length(unique(snapshots)) > 1
        @test endswith(final_output, "\n")
        @test occursin("[ Info: before progress", final_output)
        @test occursin("[ Info: after progress", final_output)
        @test occursin("\e[2K", final_output)
    end

    @testset "showvalues bar cleanup leaves no residue before logs" begin
        output = IOBuffer()
        reporter = ProcessTensors._progress_reporter(true; output=output)
        ProcessTensors._ensure_spinner!(reporter, "Stage — running")
        ProcessTensors._with_progress(reporter, "Stage — barred", 6) do update!
            for step in 1:6
                update!(step; showvalues=() -> [("time", 0.1 * step)])
            end
        end
        ProcessTensors._clear_stage!(reporter)
        println(output, "[ Info: after showvalues progress")
        final_output = String(take!(output))
        @test reporter.bar === nothing
        @test !reporter.spinner_active
        @test occursin("[ Info: after showvalues progress", final_output)
        # Durable log text after cleanup must not carry a leftover finished-bar fragment.
        info_match = match(r"\[ Info: after showvalues progress[^\n]*", final_output)
        @test info_match !== nothing
        @test !occursin("100%", info_match.match)
        @test !occursin("ETA:", info_match.match)
        @test !occursin("Time: 0:", info_match.match)
    end

    @testset "spinner refresh and cleanup" begin
        output = IOBuffer()
        reporter = ProcessTensors._progress_reporter(true; output=output)
        ProcessTensors._with_spinner(reporter, "Testing spinner") do
            sleep(0.25)
        end
        final_output = String(take!(output))
        @test !reporter.spinner_active
        @test !isempty(final_output)
        @test occursin("\r\e[2K", final_output) || occursin("Testing spinner", final_output)
    end

    @testset "header spinner survives child progress bar" begin
        output = IOBuffer()
        reporter = ProcessTensors._progress_reporter(true; output=output)
        ProcessTensors._ensure_spinner!(reporter, "Stage A — preparing")
        sleep(0.15)
        ProcessTensors._ensure_spinner!(reporter, "Stage B — exponentiating")
        sleep(0.15)
        @test reporter.spinner_active
        @test occursin("Stage B", reporter.spinner_desc)
        ProcessTensors._with_progress(reporter, "Stage C — assembling", 4) do update!
            for step in 1:4
                update!(step)
                sleep(0.05)
            end
            @test reporter.spinner_active
            @test reporter.bar !== nothing
        end
        @test reporter.bar === nothing
        @test reporter.spinner_active
        @test occursin("Stage C", reporter.spinner_desc)
        ProcessTensors._clear_stage!(reporter)
        @test !reporter.spinner_active
    end

    @testset "failure cleanup preserves exception" begin
        output = IOBuffer()
        reporter = ProcessTensors._progress_reporter(true; output=output)
        err = try
            ProcessTensors._with_progress(reporter, "Failing stage", 2) do update!
                update!(1)
                error("expected progress failure")
            end
            nothing
        catch caught
            caught
        end
        @test err isa ErrorException
        @test occursin("expected progress failure", sprint(showerror, err))
        @test reporter.bar === nothing
        @test !reporter.spinner_active
    end

    @testset "disabled reporter remains silent" begin
        output = IOBuffer()
        reporter = ProcessTensors._progress_reporter(false; output=output)
        ProcessTensors._with_progress(reporter, "Silent stage", 2) do update!
            update!(1)
            update!(2)
        end
        @test reporter.bar === nothing
        @test !reporter.spinner_active
        @test isempty(String(take!(output)))
    end

    @testset "verbose workflow logs are durable and opt-in" begin
        sites = siteinds("S=1/2", 1)
        system = spin_system(sites, OpSum() + (0.2, "Sz", 1))
        @test_logs (:info, r"Built process tensor") build_process_tensor(
            system;
            dt=0.1,
            nsteps=2,
            progress=false,
            verbose=true,
        )
    end

    @testset "evolve staged captions survive through snapshot bar" begin
        sites = siteinds("S=1/2", 1)
        system = spin_system(sites, OpSum() + (0.2, "Sz", 1))
        pt = build_process_tensor(system; dt=0.1, nsteps=3, progress=false)
        rho0 = to_dm(MPS(sites, ["Up"]))
        seq = InstrumentSeq(default=IdentityOperation(), nsteps=pt.nsteps)
        add!(seq, StatePreparation(rho0), 0)

        output = IOBuffer()
        reporter = ProcessTensors._progress_reporter(true; output=output)
        ProcessTensors._ensure_spinner!(reporter, "Evolving reduced system — starting")
        @test occursin("starting", reporter.spinner_desc)

        instruments = ProcessTensors.Instruments._create_instruments(
            pt,
            seq;
            default=IdentityOperation(),
            reporter=reporter,
            progress_desc="Evolving reduced system — materializing instruments",
        )
        @test length(instruments) == pt.nsteps + 1
        @test reporter.spinner_active
        @test occursin("materializing", reporter.spinner_desc)

        ProcessTensors._ensure_spinner!(reporter, "Evolving reduced system — preparing trajectory")
        @test occursin("preparing", reporter.spinner_desc)

        ProcessTensors._with_progress(
            reporter,
            "Evolving reduced system — computing snapshots",
            pt.nsteps,
        ) do update!
            for k in 1:pt.nsteps
                update!(k)
            end
            @test reporter.spinner_active
            @test reporter.bar !== nothing
        end
        @test reporter.bar === nothing
        @test occursin("snapshots", reporter.spinner_desc)
        ProcessTensors._clear_stage!(reporter)

        @test_logs (:info, r"Evolved reduced system") evolve(
            pt, rho0; progress=false, verbose=true,
        )
    end
end
