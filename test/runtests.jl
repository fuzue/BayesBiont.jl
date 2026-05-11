using BayesBiont
using Kinbiont: GrowthData, MODEL_REGISTRY
using Distributions
using Random
using Test
using Aqua

include("utils.jl")

@testset "BayesBiont.jl" begin
    @testset "Aqua" begin
        Aqua.test_all(BayesBiont; ambiguities=false, deps_compat=(check_extras=false,))
    end

    @testset "API surface" begin
        @test isdefined(BayesBiont, :bayesfit)
        @test isdefined(BayesBiont, :bayesian_fit)
        @test BayesBiont.bayesfit === BayesBiont.bayesian_fit
        @test isdefined(BayesBiont, :BayesianModelSpec)
        @test isdefined(BayesBiont, :BayesFitOptions)
        @test isdefined(BayesBiont, :BayesianCurveFitResult)
        @test isdefined(BayesBiont, :BayesianGrowthFitResults)
        @test isdefined(BayesBiont, :group_from_labels)
    end

    @testset "group_from_labels" begin
        data = GrowthData(rand(4, 5), collect(0.0:0.25:1.0), ["WT_1", "WT_2", "mut_1", "mut_2"])
        @test group_from_labels(data) == ["WT", "WT", "mut", "mut"]
    end

    @testset "v0.1 reserves group= kwarg" begin
        times, y, _ = synthetic_gompertz()
        data = GrowthData(reshape(y, 1, :), times, ["c1"])
        spec = BayesianModelSpec([MODEL_REGISTRY["NL_Gompertz"]])
        @test_throws ArgumentError bayesfit(data, spec; group=["A"])
    end

    @testset "lognormal requires positive data" begin
        times, y, _ = synthetic_gompertz()
        y_bad = copy(y); y_bad[1] = -0.01
        data = GrowthData(reshape(y_bad, 1, :), times, ["c1"])
        spec = BayesianModelSpec([MODEL_REGISTRY["NL_Gompertz"]])
        @test_throws ArgumentError bayesfit(data, spec, BayesFitOptions(likelihood=:lognormal))
    end

    @testset "Gompertz recovery" begin
        times, observed, truth = synthetic_gompertz(seed=11)
        data = GrowthData(reshape(observed, 1, :), times, ["g1"])
        spec = BayesianModelSpec([MODEL_REGISTRY["NL_Gompertz"]])
        opts = BayesFitOptions(n_chains=2, n_warmup=400, n_samples=400, rng_seed=11)

        results = bayesfit(data, spec, opts)
        @test results isa BayesianGrowthFitResults
        @test length(results) == 1

        r = results[1]
        @test r isa BayesianCurveFitResult
        @test r.label == "g1"
        @test r.times == times

        # Posterior means recover ground truth within 15% (loose — 400 samples × 2 chains).
        @test isapprox(mean(r.N_max),       truth.N_max;       rtol=0.15)
        @test isapprox(mean(r.growth_rate), truth.growth_rate; rtol=0.20)
        @test isapprox(mean(r.lag),         truth.lag;         rtol=0.15)
    end
end
