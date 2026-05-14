# [Models](@id models)

```@contents
Pages = ["index.md"]
Depth = 2
```

BayesBiont fits **the same model definitions Kinbiont uses**. Pull them from `Kinbiont.MODEL_REGISTRY` and pass them to a `BayesianModelSpec`.

## Closed-form NL models

The full list is in `Kinbiont.MODEL_REGISTRY`. The ones most commonly used:

| Registry key | Function | Parameters |
|---|---|---|
| `"NL_Gompertz"` | `N_max * exp(-exp(-r * (t - λ)))` | `N_max, growth_rate, lag` |
| `"NL_logistic"` | `K / (1 + (K/N_0 - 1) * exp(-r * t))` | `N_max, growth_rate, lag` (Kinbiont labels — see note below) |
| `"NL_exponential"` | `N_0 * exp(r * t)` | `N_0, growth_rate` |
| `"NL_Richards"` | Richards generalised logistic | 4 params |
| `"NL_Bertalanffy"` | Von Bertalanffy | 4 params |
| `"NL_Morgan"` | Morgan-Mercer-Flodin | 4 params |
| `"NL_Weibull"` | Weibull | 4 params |

### Naming caveat for NL_logistic

Kinbiont labels the three positional parameters of `NL_logistic` as `[N_max, growth_rate, lag]`, but the underlying function `K / (1 + (K/N_0 - 1) * exp(-r * t))` is actually parameterised as `[K, N_0, r]`. BayesBiont passes the names through verbatim; `r.growth_rate` therefore returns posterior samples of `N_0` (the initial-population parameter), and `r.lag` returns the rate. This is upstream and tracked in Kinbiont; until it's fixed, treat the **positional semantics**, not the label, as authoritative for `NL_logistic`.

## ODE models

ODE models from Kinbiont's `MODEL_REGISTRY` (e.g. `"aHPM"`, `"HPM"`, `"HPM_3_death"`, `"logistic"`, `"gompertz"`, `"baranyi_richards"`) work with `bayesfit` too. The ODE is integrated inside the Turing model at each NUTS step.

```julia
spec = BayesianModelSpec([MODEL_REGISTRY["aHPM"]])
opts = BayesFitOptions(n_chains=2, n_warmup=500, n_samples=500)
post = bayesfit(data, spec, opts)
```

Default observation convention for multi-state ODE models: the predicted OD/cell concentration at each timepoint is the **sum of all state variables** (matches Kinbiont's `sum_fin` convention). Initial condition is `u0 = [y[1], 0, ..., 0]` — all observed mass placed in the first state, others at zero.

### ODE + ReverseDiff (optional speedup)

For hierarchical fits with many curves, the ForwardDiff backend can become expensive (it scales as the square of the parameter count). ReverseDiff with a compiled tape is roughly constant per-parameter:

```julia
using SciMLSensitivity        # required for ODE + reverse-mode AD
opts = BayesFitOptions(adbackend=:reversediff, n_chains=2, n_warmup=500, n_samples=500)
post = bayesfit(data, spec, opts)
```

Measured: 6-well hierarchical aHPM took ~21 min with ReverseDiff vs ~40+ min with ForwardDiff on the same data.

## Likelihoods

Set with `BayesFitOptions(likelihood=...)`:

| Value | What it is | When to use |
|---|---|---|
| `:lognormal` (default) | Multiplicative log-normal noise on OD. Equivalent to constant relative error. | OD data; matches Kinbiont's `RE` loss semantically. |
| `:normal` | Additive Normal noise. | Linear-scale data; when noise is independent of magnitude. |
| `:proportional` | Heteroscedastic Normal with `σ_i = σ · pred_i`. | Pure relative-error noise model; behaves like `:lognormal` but on raw scale. |

The `:lognormal` default requires strictly positive data. If your input has zeros or negatives (e.g. blank-subtraction artefacts), either preprocess with Kinbiont's `correct_negatives=true` or set `likelihood=:normal`.

## Custom models

Any `Kinbiont.NLModel` or `Kinbiont.ODEModel` you build yourself works:

```julia
my_curve(p, t) = p[1] .* exp.(p[2] .* t .- p[3] .* t.^2)
my_model = NLModel(
    "my_quadratic_exp",
    my_curve,
    ["scale", "rate", "saturation"],
    nothing,
)
spec = BayesianModelSpec([my_model])
post = bayesfit(data, spec)    # falls back to empirical priors from model.guess (or prior medians if no guess)
```

You'll typically want to supply explicit priors for custom models — see [Priors](@ref priors).
