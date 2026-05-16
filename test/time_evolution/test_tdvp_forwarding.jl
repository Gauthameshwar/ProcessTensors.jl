using ProcessTensors
using ITensors
using Test

@testset "tdvp.jl forwarding API" begin
    sites = siteinds("S=1/2", 2)
    liouv = liouv_sites(sites)
    os_H = OpSum()
    os_H += 0.5, "Sz", 1
    ρ0 = to_liouville(to_dm(MPS(sites, ["Up", "Up"])); sites=liouv)
    L_mpo = MPO_Liouville(os_H, liouv; jump_ops=Tuple{Number, String, Int}[])

    tdvp_kwargs = (; time_step=0.05, nsite=1, maxdim=16, cutoff=1e-10, outputlevel=0)

    state_timed = tdvp(L_mpo, 0.05, ρ0; tdvp_kwargs...)
    @test state_timed isa MPS{Liouville}

    state_core_timed = tdvp(L_mpo.core, 0.05, ρ0; tdvp_kwargs...)
    @test state_core_timed isa MPS{Liouville}
end
