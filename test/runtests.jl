using ProcessTensors
using Test
using Aqua

@testset "Aqua.jl: Code quality checks" begin
    Aqua.test_all(ProcessTensors; piracies=false)
end

include(joinpath(@__DIR__, "liouvillian", "liouvillian_hilbert_roundtrip.jl"))

include(joinpath(@__DIR__, "liouvillian", "single_spin_analytical.jl"))

include(joinpath(@__DIR__, "liouvillian", "liouvillian_methods.jl"))

if Base.find_package("QuantumOptics") !== nothing
    include(joinpath(@__DIR__, "liouvillian", "spinmpo_vs_qo.jl"))
    include(joinpath(@__DIR__, "liouvillian", "bosonmpo_vs_qo.jl"))
    nothing
else
    @info "Skipping QuantumOptics comparison tests because QuantumOptics is not installed."
end

include(joinpath(@__DIR__, "time_evolution", "tebd_validation.jl"))
include(joinpath(@__DIR__, "time_evolution", "tebd_tfim_benchmarks.jl"))
include(joinpath(@__DIR__, "time_evolution", "tdvp_tfim_benchmarks.jl"))

include(joinpath(@__DIR__, "systems", "test_instruments.jl"))
