using MicroGrad
using Test

@testset "MicroGrad tests" begin

    @testset "engine tests" begin
        include("engine_tests.jl")
    end
end