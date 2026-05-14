using ProcessTensors
using ITensors
using Test

if !isdefined(Main, :liouville_state_to_dense)
    include(joinpath(@__DIR__, "time_evolution", "tebd_test_utils.jl"))
end

@testset "process_tensor.jl: single-site rebuild API" begin
    @testset "build_process_tensor uses explicit coupling_site::Index" begin
        s = siteinds("S=1/2", 2)
        H = OpSum()
        H += 0.3, "Sz", 1
        H += -0.1, "Sz", 2
        system = spin_system(s, H)

        pt = build_process_tensor(system, system.sites[2]; dt=0.1, nsteps=3)
        @test pt isa ProcessTensor
        @test pt.coupling_site == system.sites[2]
        @test length(pt.core) == pt.nsteps == 3
        @test length(output_sites(pt, 0)) == 1
        @test length(input_sites(pt, 2)) == 1

        @test_throws ArgumentError build_process_tensor(system, Index(dim(system.sites[1])); dt=0.1, nsteps=3)
    end

    @testset "coupling_times / coupling_sites resolve stable PT legs" begin
        s = siteinds("S=1/2", 1)
        system = spin_system(s, OpSum())
        pt = build_process_tensor(system, system.sites[1]; dt=0.2, nsteps=4)
        out1, in2 = coupling_times(pt, 2)
        @test length(in2) == 1
        @test length(out1) == 1
        @test plev(only(in2)) == 1
        @test plev(only(out1)) == 0
        @test tag_value(only(in2), "tstep=") == "2"
        @test tag_value(only(out1), "tstep=") == "1"

        # Stable object identity across repeated calls (no fresh-mint index drift).
        out1_again, in2_again = coupling_times(pt, 2)
        @test only(out1_again) == only(out1)
        @test only(in2_again) == only(in2)

        in2b, out1b = coupling_sites(pt, 2)
        @test only(in2b) == only(in2)
        @test only(out1b) == only(out1)
    end

    @testset "build_process_tensor rejects multi-mode AbstractBath (v1 single-mode only)" begin
        s = siteinds("S=1/2", 1)
        e1 = siteinds("S=1/2", 1)
        e2 = siteinds("S=1/2", 1)
        L_sys = liouv_sites(s)
        L1 = liouv_sites(e1)
        L2 = liouv_sites(e2)
        H_sys = OpSum()
        H_sys += 0.2, "Sz", 1
        sys = spin_system(s, H_sys)

        ρ1 = to_liouville(to_dm(MPS(e1, ["Up"])); sites=L1)
        H_env = OpSum()
        H_env += 1.0, "Sx", 1
        m1 = SpinMode(L1, H_env, ρ1)
        ρ2 = to_liouville(to_dm(MPS(e2, ["Up"])); sites=L2)
        m2 = SpinMode(L2, H_env, ρ2)

        coupling = OpSum()
        coupling += 0.05, "Sz", 1, "Sz", 2
        coupling += 0.03, "Sz", 1, "Sz", 3
        bad_bath = spin_bath([m1, m2]; coupling=coupling)
        @test_throws ArgumentError build_process_tensor(sys, only(L_sys); environment=bad_bath, dt=0.05, nsteps=2)
    end

    @testset "single-site instrument validation" begin
        s = siteinds("S=1/2", 1)
        psi0 = MPS(s, ["Up"])
        bad_input = [prime(s[1]), prime(s[1])]
        bad_output = [s[1], s[1]]
        op_sz = OpSum()
        op_sz += 1.0, "Sz", 1

        @test_throws ArgumentError StatePreparation(psi0, bad_input; leg_plev=1)
        @test_throws ArgumentError ObservableMeasurement(op_sz, bad_output; leg_plev=0)
        @test_throws ArgumentError TraceOut(bad_output; leg_plev=0)
        @test_throws ArgumentError IdentityOperation(bad_input, bad_output)
    end

    @testset "markovian fallback evolve still works without environment" begin
        s = siteinds("S=1/2", 1)
        H = OpSum()
        H += 0.5, "Sz", 1
        L = OpSum()
        L += 0.2, "S-", 1
        system = spin_system(s, H; jump_ops=[L])
        pt = build_process_tensor(system; dt=0.05, nsteps=3)
        psi0 = MPS(s, ["Up"])
        rho0 = to_liouville(to_dm(psi0); sites=system.sites)

        trajectory = evolve(pt, psi0)
        manual = tebd_trajectory(rho0, H, 0.05, 3; jump_ops=[L], maxdim=32, cutoff=1e-12, order=2)

        @test trajectory.times ≈ [0.0, 0.05, 0.1] atol=1e-12
        @test length(trajectory.states_liouville) == 3
    end
end
