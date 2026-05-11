# Quickstart: fit a Gompertz curve on synthetic data, print posterior summary.
# Run from the BayesBiont repo root with: julia --project=. examples/quickstart.jl

using BayesBiont, Kinbiont, Statistics, Random

Random.seed!(42)

times = collect(0.0:0.25:24.0)
truth = (N_max=1.0, growth_rate=0.4, lag=5.0)
clean = truth.N_max .* exp.(-exp.(-truth.growth_rate .* (times .- truth.lag)))
obs   = clean .* exp.(0.05 .* randn(length(times)))

data = GrowthData(reshape(obs, 1, :), times, ["well1"])
spec = BayesianModelSpec([MODEL_REGISTRY["NL_Gompertz"]])
opts = BayesFitOptions(n_chains=2, n_warmup=400, n_samples=400, rng_seed=42)

post = bayesfit(data, spec, opts)
r = post[1]

println("Posterior summary (truth in parentheses):")
for (name, t) in zip([:N_max, :growth_rate, :lag], [truth.N_max, truth.growth_rate, truth.lag])
    samples = getproperty(r, name)
    lo, hi  = quantile(samples, [0.025, 0.975])
    println("  $(rpad(string(name), 12))  mean=$(round(mean(samples); digits=3))  " *
            "95% CI=[$(round(lo; digits=3)), $(round(hi; digits=3))]  truth=$t")
end

ppc = posterior_predict(r; n_draws=200)
println("\nPPC mean at t=$(times[end]): $(round(mean(ppc[:, end]); digits=3))  obs=$(round(obs[end]; digits=3))")
