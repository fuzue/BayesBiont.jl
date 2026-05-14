# BayesBiont.jl

**BayesBiont.jl** is the Bayesian companion to [Kinbiont.jl](https://github.com/pinheiroGroup/Kinbiont.jl). It adds calibrated uncertainty quantification, hierarchical pooling across replicates, and principled model comparison to the same growth-curve models the field already uses.

It is additive, not competitive: BayesBiont consumes Kinbiont's `GrowthData` container and curve definitions directly, and returns posterior distributions where Kinbiont returns point estimates.

## At a glance

| Kinbiont | BayesBiont |
|---|---|
| Maximum-likelihood / least-squares point estimates | Posterior distributions (NUTS sampling via Turing) |
| Delta-method or bootstrap CIs | Calibrated credible intervals |
| Replicates fit independently | Hierarchical pooling via `group=` |
| AICc model selection | LOO / WAIC + `compare()` for ELPD differences |
| No self-diagnostic | Pareto-k flags when LOO is unreliable |

The same `GrowthData(curves, times, labels)` you'd build for Kinbiont's `kinbiont_fit` goes straight into BayesBiont's `bayesfit` — usually <10 lines of code change.

## New to BayesBiont?

Start with [Install](@ref install), then [Quick start](@ref quickstart). The other sections cover specific capabilities:

```@contents
Pages = [
    "01_install/index.md",
    "02_quickstart/index.md",
    "03_models/index.md",
    "04_priors/index.md",
    "05_hierarchical/index.md",
    "06_comparison/index.md",
    "07_diagnostics/index.md",
    "api.md",
]
Depth = 2
```

## When to reach for BayesBiont

- You compare strains, treatments, or conditions and want calibrated probability statements about differences (`P(μ_A > μ_B)`) rather than ad-hoc t-tests on fitted point estimates.
- You have replicates per condition and want to pool information *correctly* across them (hierarchical Bayes), not average-then-fit.
- You want to know whether a fitted ODE model (aHPM, HPM, etc.) is actually appropriate for your data — `loo()` + Pareto-k tells you when the answer is "no".
- You're producing posterior predictive bands on growth-curve fits, not just point fits.

## When you don't need BayesBiont

- You want a quick point estimate of growth rate from one curve — stick with [Kinbiont](https://github.com/pinheiroGroup/Kinbiont.jl), it's faster.
- You don't have replicates and don't care about uncertainty — point estimates are fine.
- You're doing exploratory data inspection — Kinbiont's NL fits return in milliseconds; BayesBiont's NUTS runs take seconds to minutes per curve.

## Citing

If you use BayesBiont in a publication, please cite both BayesBiont and the underlying Kinbiont paper:

> Angaroni F. et al. *Kinbiont.jl: a flexible Julia toolkit for kinetic modelling of microbial systems*. (Kinbiont reference.)

## License

MIT. Maintained by [Fuzue Tech](https://fuzue.tech).
