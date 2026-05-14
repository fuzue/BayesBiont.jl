# [Quick start](@id quickstart)

```@contents
Pages = ["index.md"]
Depth = 3
```

Fit a Bayesian posterior to a single growth curve in under 15 lines.

## Synthetic Gompertz example

```julia
using BayesBiont, Kinbiont, Statistics, Random

Random.seed!(42)
times = collect(0.0:0.25:24.0)
truth = (N_max=1.0, growth_rate=0.4, lag=5.0)
clean = truth.N_max .* exp.(-exp.(-truth.growth_rate .* (times .- truth.lag)))
obs   = clean .* exp.(0.05 .* randn(length(times)))   # 5 % multiplicative noise

data = GrowthData(reshape(obs, 1, :), times, ["well1"])
spec = BayesianModelSpec([MODEL_REGISTRY["NL_Gompertz"]])
opts = BayesFitOptions(n_chains=2, n_warmup=400, n_samples=400)

post = bayesfit(data, spec, opts)
r    = post[1]

println("growth_rate = $(mean(r.growth_rate))  95% CI = $(quantile(r.growth_rate, [0.025, 0.975]))")
# growth_rate = 0.4019  95% CI = [0.398, 0.409]
```

On this 96-point synthetic curve, BayesBiont recovers all three parameters to ~2 % with tight 95 % credible intervals.

## What just happened

| Step | What it did |
|---|---|
| `GrowthData(reshape(obs, 1, :), times, ["well1"])` | Same container Kinbiont uses — one row per curve. |
| `BayesianModelSpec([MODEL_REGISTRY["NL_Gompertz"]])` | Pick a Kinbiont model; BayesBiont auto-derives priors from `model.guess()`. |
| `BayesFitOptions(...)` | All NUTS knobs (chains, warmup, samples, target_accept, max_treedepth) with sensible defaults. |
| `bayesfit(data, spec, opts)` | Returns `BayesianGrowthFitResults` (one row per input curve). |
| `r.growth_rate` | Field-access shortcut for posterior samples — works for any parameter in the model. |

## Posterior predictive

```julia
ppc = posterior_predict(r; n_draws=300)            # 300 × n_timepoints
ppc_mean = vec(mean(ppc; dims=1))
ppc_band = mapslices(c -> quantile(c, [0.025, 0.975]), ppc; dims=1)
```

## Loading real data from CSV

The data container is shared with Kinbiont. The Kinbiont column convention (first column = times, remaining columns = wells) works directly:

```julia
data = GrowthData("plate.csv")            # 96-well plate
spec = BayesianModelSpec([MODEL_REGISTRY["NL_Gompertz"]])
post = bayesfit(data, spec)               # fits each well independently
```

Each row of `post` is a `BayesianCurveFitResult` carrying the posterior chains. For hierarchical pooling across replicates of the same biological condition, see [Hierarchical](@ref hierarchical).

## Next steps

- [Models](@ref models) — what curves (NL closed-form + ODE) BayesBiont supports
- [Priors](@ref priors) — curated default priors vs user-supplied priors
- [Hierarchical](@ref hierarchical) — pooling across replicates + `contrast()`
- [Model comparison](@ref comparison) — LOO / WAIC / `compare()`
