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

## Quick start (planned API)

```julia
using BayesBiont, Kinbiont

data = GrowthData("plate.csv")
spec = BayesianModelSpec([MODEL_REGISTRY["NL_Gompertz"]])  # auto-derived priors

post = bayesfit(data, spec)
post[1].growth_rate    # posterior samples for the first curve's μ
mean(post[1].growth_rate), quantile(post[1].growth_rate, [0.025, 0.975])
```

## Roadmap

- **v0.1** — single-curve fits (logistic, Gompertz), NUTS, `:lognormal`/`:normal` likelihoods
- **v0.2** — hierarchical pooling, Baranyi + Richards, ADVI fast path, ReverseDiff
- **v0.3** — WAIC / LOO / Bayes factors, posterior predictive checks, Student-t likelihood
- **v1.0** — docs, paper-ready examples, Julia registry release

## License

MIT
