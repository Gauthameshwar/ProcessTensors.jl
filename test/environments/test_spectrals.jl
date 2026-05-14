using ProcessTensors
using ProcessTensors.Spectrals: OhmicSpectralDensity, LorentzianSpectralDensity,
                              ohmic_sd, lorentzian_sd
using Test

@testset "spectrals.jl: spectral density structs and helper constructors" begin
    ohm = ohmic_sd()
    lor = lorentzian_sd()

    @test ohm isa OhmicSpectralDensity
    @test lor isa LorentzianSpectralDensity
    @test ohm.alpha == 1.0
    @test ohm.wc == 1.0
    @test ohm.s == 1.0
    @test lor.lambda == 1.0
    @test lor.gamma == 1.0
    @test lor.omega0 == 0.0

    ohm2 = ohmic_sd(alpha=2, wc=3.5, s=0.75)
    lor2 = lorentzian_sd(lambda=1.2, gamma=0.8, omega0=2)

    @test ohm2 isa OhmicSpectralDensity{Float64}
    @test lor2 isa LorentzianSpectralDensity{Float64}
    @test ohm2.alpha == 2.0
    @test ohm2.wc == 3.5
    @test ohm2.s == 0.75
    @test lor2.lambda == 1.2
    @test lor2.gamma == 0.8
    @test lor2.omega0 == 2.0
end
