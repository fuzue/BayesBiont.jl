using BayesBiont
using Test
using Aqua

@testset "BayesBiont.jl" begin
    @testset "Aqua" begin
        Aqua.test_all(BayesBiont; ambiguities=false)
    end
end
