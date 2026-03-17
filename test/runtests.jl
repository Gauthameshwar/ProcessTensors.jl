using ProcessTensors
using Test
using Aqua

@testset "ProcessTensors.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(ProcessTensors)
    end
    # Write your tests here.
end
