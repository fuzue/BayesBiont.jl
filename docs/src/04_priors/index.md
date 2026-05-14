# [Priors](@id priors)

```@contents
Pages = ["index.md"]
Depth = 2
```

Priors in BayesBiont are explicit, optional, and biology-grounded by default. The package ships with curated priors for canonical models and falls back to empirical priors derived from `Kinbiont.NLModel.guess(...)` for anything else.

## Default behaviour

Pass a `BayesianModelSpec` without `priors=` and BayesBiont picks:

1. **Curated entry from `DEFAULT_PRIORS`** if your model name is registered (e.g. `"NL_Gompertz"`, `"NL_logistic"`, `"aHPM"`).
2. **Empirical fallback** — `LogNormal(log(guess_i), 1.0)` per parameter, where `guess_i` comes from `model.guess(data_matrix)`. A 95 % CI of roughly `×7` either way.

You can inspect the curated registry:

```julia
using BayesBiont
BayesBiont.DEFAULT_PRIORS["NL_Gompertz"]
# (
#     N_max       = LogNormal(log(1.0), 0.7),   # ~0.25 .. 4 OD
#     growth_rate = LogNormal(log(0.5), 0.8),   # ~0.10 .. 2.4 hr⁻¹
#     lag         = LogNormal(log(5.0), 1.0),   # ~0.7  .. 35 hr
# )
```

The curated priors are calibrated for **plate-reader OD data on typical bacteria**. If your scale is different (cell counts, slow-growing organisms, dense cultures) supply explicit priors.

## Supplying your own priors

```julia
custom_priors = (
    N_max       = LogNormal(log(2.0), 0.4),     # tighter; expecting ~2 OD plateau
    growth_rate = LogNormal(log(0.8), 0.5),
    lag         = LogNormal(log(3.0), 0.4),
)
spec = BayesianModelSpec(
    [MODEL_REGISTRY["NL_Gompertz"]];
    priors = [custom_priors],
    sigma_prior = Exponential(0.05),            # 5 % expected noise
)
```

The `priors` field is a vector — one prior `NamedTuple` per model in the spec (currently only the first model is fit; future versions will support multi-model spec for `compare()`).

## What the priors do internally

For positive parameters with `LogNormal` priors, BayesBiont samples in **log-space** internally (Normal on `log(p)`, deterministic transform to `p`). For hierarchical fits, this becomes a non-centered reparameterisation across replicates, which sidesteps Neal's funnel and dramatically improves NUTS efficiency. Users always see native-scale samples in the output `Chains` — the log-space trick is invisible.

## Tightness as a diagnostic

If you see the posterior staying very close to your prior centre regardless of data, the data isn't informative about that parameter. This is a signal worth heeding:

- For identifiability problems on `lag`-like parameters with short experiments, the lag prior dominates — narrow your prior using biological knowledge.
- For poorly-constrained parameters in ODE models (e.g. `exit_lag_rate` in aHPM), expect wide posteriors unless your data sampling captures the relevant phase.
- The Pareto-k diagnostic in [Diagnostics](@ref diagnostics) is the formal version of this check.

## Public API

- [`BayesianModelSpec`](@ref BayesBiont.BayesianModelSpec)
- [`DEFAULT_PRIORS`](@ref BayesBiont.DEFAULT_PRIORS)
- [`default_priors`](@ref BayesBiont.default_priors)
- [`empirical_priors`](@ref BayesBiont.empirical_priors)
