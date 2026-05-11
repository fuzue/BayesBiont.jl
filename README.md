# BayesBiont.jl

Bayesian companion to [Kinbiont.jl](https://github.com/pinheiroGroup/KinBiont.jl). Calibrated uncertainty quantification, hierarchical pooling across replicates, and principled model comparison for microbial growth curves.

> **Status:** v0.1.0-DEV — early development. APIs may change before v1.0.

## What it adds over Kinbiont

| | Kinbiont | BayesBiont |
|---|---|---|
| Parameter estimates | Point (MLE) | Posterior distributions |
| Uncertainty | Delta-method / bootstrap | Calibrated credible intervals |
| Replicate handling | Independent fits | Hierarchical pooling (v0.2) |
| Model comparison | AICc | WAIC, LOO-PSIS, Bayes factors (v0.3) |
| High-throughput plates | Fast (optimization) | ADVI fast path (v0.2) |

BayesBiont reuses Kinbiont's curve definitions and `GrowthData` container — it's additive, not competitive.

## Quick start

```julia
using BayesBiont, Kinbiont, Statistics

# Synthetic Gompertz curve with 5% multiplicative noise.
times = collect(0.0:0.25:24.0)
truth = (N_max=1.0, growth_rate=0.4, lag=5.0)
clean = truth.N_max .* exp.(-exp.(-truth.growth_rate .* (times .- truth.lag)))
obs   = clean .* exp.(0.05 .* randn(length(times)))

data = GrowthData(reshape(obs, 1, :), times, ["well1"])
spec = BayesianModelSpec([MODEL_REGISTRY["NL_Gompertz"]])     # auto-derived priors
opts = BayesFitOptions(n_chains=2, n_warmup=400, n_samples=400)

post = bayesfit(data, spec, opts)

r = post[1]
mean(r.growth_rate), quantile(r.growth_rate, [0.025, 0.975])

# Posterior predictive bands on the original time grid.
ppc = posterior_predict(r; n_draws=200)
```

To load real data from a CSV (Kinbiont column convention — first column times,
remaining columns one curve each), pass a path to `GrowthData`:

```julia
data = GrowthData("plate.csv")
```

## Roadmap

- **v0.1** — single-curve fits (logistic, Gompertz), NUTS, `:lognormal`/`:normal` likelihoods
- **v0.2** — hierarchical pooling, Baranyi + Richards, ADVI fast path, ReverseDiff
- **v0.3** — WAIC / LOO / Bayes factors, posterior predictive checks, Student-t likelihood
- **v1.0** — docs, paper-ready examples, Julia registry release

## License

MIT
