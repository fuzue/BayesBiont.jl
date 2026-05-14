# [Model comparison](@id comparison)

```@contents
Pages = ["index.md"]
Depth = 2
```

Once you have two fits, `compare(r1, r2)` tells you which one the data prefers — with calibrated uncertainty on the difference and a self-diagnostic that flags when the comparison is unreliable.

## LOO and WAIC

The [`loo`](@ref BayesBiont.loo) and [`waic`](@ref BayesBiont.waic) functions return ELPD estimates for a single fit:

```julia
r = post[1]    # a BayesianCurveFitResult

w = waic(r)
# (elpd = 230.52, p_eff = 3.46, n_obs = 97)

l = loo(r)
# (elpd = 230.52, se = 16.98, n_obs = 97, pareto_k_max = 0.624, elpd_pointwise = [...])
```

`elpd` is the **expected log pointwise predictive density** — a Bayesian analogue of out-of-sample log-likelihood. Higher is better. `loo` uses Pareto-smoothed importance sampling (via `PSIS.jl`) to estimate it from the posterior at no additional sampling cost.

### The Pareto-k diagnostic

`pareto_k_max` is the largest Pareto-k value across observations. The standard interpretation:

| `k` | Interpretation |
|---|---|
| < 0.5 | LOO estimate is reliable. |
| 0.5 – 0.7 | LOO is approximately reliable. |
| 0.7 – 1.0 | Importance sampling is unreliable for some observation; treat ELPD with caution. |
| > 1.0 | LOO cannot be trusted on this data; the model is likely badly misspecified. |

The diagnostic is **self-firing** — no separate calibration check needed. In our own validation: fitting Gompertz to aHPM-generated data produces `pareto_k_max ≈ 1.19`, correctly signalling that this model is the wrong one.

## Comparing two models

The [`compare`](@ref BayesBiont.compare) function computes the ELPD difference between two fits:

```julia
r_correct  = bayesfit(data, BayesianModelSpec([MODEL_REGISTRY["aHPM"]]))[1]
r_wrong    = bayesfit(data, BayesianModelSpec([MODEL_REGISTRY["NL_Gompertz"]]))[1]

cmp = compare(r_correct, r_wrong)
# (elpd_diff = +117.6, se_diff = 10.0, favours = "aHPM (>2 SE)")
```

`elpd_diff = elpd(r1) - elpd(r2)` with its standard error. Conventional reading: if `|elpd_diff| > 2 * se_diff`, the difference is meaningful. The `favours` field applies that threshold and reports the verdict in plain text.

## Pointwise log-likelihoods

For users who want to do their own ELPD analysis or write their own diagnostics, [`pointwise_loglik`](@ref BayesBiont.pointwise_loglik) returns an `n_samples × n_obs` matrix. You can pipe this into any standard model-comparison machinery.

## What this gives you that Kinbiont's AICc doesn't

| Kinbiont | BayesBiont |
|---|---|
| AICc per model | Full ELPD posterior |
| No diagnostic on whether AICc is reliable | Pareto-k flags when LOO is unreliable |
| Picks the best AICc model | Reports ELPD difference + SE so you can judge significance |
| Asymptotic; needs N ≫ params | Bayesian; works for small N too |

If your goal is "which of these N candidate models should I report?", Kinbiont's AICc is faster. If your goal is "is the difference between these two fits statistically meaningful?", `compare()` answers that with an uncertainty estimate.
