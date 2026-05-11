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

        @testset "propertynames + getproperty" begin
            pn = propertynames(r)
            @test :label in pn
            @test :chains in pn
            @test :times in pn
            @test :N_max in pn
            @test :growth_rate in pn
            @test :lag in pn
            @test :σ in pn
            @test_throws ArgumentError r.does_not_exist
        end

        @testset "posterior_predict" begin
            ppc = posterior_predict(r; n_draws=50)
            @test size(ppc) == (50, length(times))
            ppc_mean = vec(mean(ppc, dims=1))
            # Posterior-predictive mean curve sits near the observed data.
            @test isapprox(ppc_mean, observed; rtol=0.30)
        end

        @testset "show printers" begin
            buf = IOBuffer()
            show(IOContext(buf, :displaysize => (24, 80)), MIME("text/plain"), results)
            outer = String(take!(buf))
            @test occursin("BayesianGrowthFitResults", outer)
            @test occursin("g1", outer)
            @test occursin("NL_Gompertz", outer)

            buf = IOBuffer()
            show(IOContext(buf, :displaysize => (24, 80)), MIME("text/plain"), r)
            inner = String(take!(buf))
            @test occursin("BayesianCurveFitResult", inner)
            @test occursin("N_max", inner)
            @test occursin("95% CI", inner)
        end
    end

    @testset "user-supplied priors are honoured" begin
        # Delta-like prior centred *off* the truth — if priors flow through, the
        # posterior must concentrate near the prior centre regardless of the data.
        times, observed, truth = synthetic_gompertz(seed=11)
        data = GrowthData(reshape(observed, 1, :), times, ["g1"])
        custom_priors = (
            N_max = LogNormal(log(0.5), 1e-3),         # essentially a point mass at 0.5
            growth_rate = LogNormal(log(truth.growth_rate), 0.5),
            lag = LogNormal(log(truth.lag), 0.5),
        )
        spec = BayesianModelSpec([MODEL_REGISTRY["NL_Gompertz"]]; priors=[custom_priors])
        opts = BayesFitOptions(n_chains=1, n_warmup=200, n_samples=200, rng_seed=11)

        r = bayesfit(data, spec, opts)[1]
        # Truth is 1.0, prior pins 0.5 with effectively-zero width.
        @test isapprox(mean(r.N_max), 0.5; rtol=0.02)
    end

    @testset ":normal likelihood recovers Gompertz" begin
        # Additive noise version — switch likelihood, expect comparable recovery.
        Random.seed!(7)
        times = collect(0.0:0.25:24.0)
        truth = (N_max=1.0, growth_rate=0.4, lag=5.0, σ=0.03)
        clean = truth.N_max .* exp.(-exp.(-truth.growth_rate .* (times .- truth.lag)))
        observed = clean .+ truth.σ .* randn(length(times))
        data = GrowthData(reshape(observed, 1, :), times, ["g_norm"])
        spec = BayesianModelSpec([MODEL_REGISTRY["NL_Gompertz"]])
        opts = BayesFitOptions(likelihood=:normal,
                               n_chains=2, n_warmup=400, n_samples=400, rng_seed=7)

        r = bayesfit(data, spec, opts)[1]
        @test isapprox(mean(r.N_max),       truth.N_max;       rtol=0.10)
        @test isapprox(mean(r.growth_rate), truth.growth_rate; rtol=0.20)
        @test isapprox(mean(r.lag),         truth.lag;         rtol=0.15)
    end

    @testset "DEFAULT_PRIORS registry" begin
        @test haskey(BayesBiont.DEFAULT_PRIORS, "NL_Gompertz")
        @test haskey(BayesBiont.DEFAULT_PRIORS, "NL_logistic")
        gomp = BayesBiont.DEFAULT_PRIORS["NL_Gompertz"]
        @test gomp.N_max isa LogNormal
        @test gomp.growth_rate isa LogNormal
        @test gomp.lag isa LogNormal
    end

    @testset "logistic recovery" begin
        times, observed, truth = synthetic_logistic(seed=23)
        data = GrowthData(reshape(observed, 1, :), times, ["l1"])
        spec = BayesianModelSpec([MODEL_REGISTRY["NL_logistic"]])
        opts = BayesFitOptions(n_chains=2, n_warmup=400, n_samples=400, rng_seed=23)

        results = bayesfit(data, spec, opts)
        r = results[1]

        # Kinbiont's NL_logistic exposes its three params positionally; the registry
        # labels are misleading (see Kinbiont bug note). Names below map to
        # [K, N_0, r] via Kinbiont's declared param_names.
        @test isapprox(mean(r.N_max),       truth.K;   rtol=0.15)
        @test isapprox(mean(r.growth_rate), truth.N_0; rtol=0.40)  # N_0 weakly identified
        @test isapprox(mean(r.lag),         truth.r;   rtol=0.20)
    end
end
