# [Diagnostics](@id diagnostics)

```@contents
Pages = ["index.md"]
Depth = 2
```

The single biggest practical advantage of a properly-priored Bayesian fit is **self-diagnosing uncertainty**: when the model is wrong, the workflow tells you. BayesBiont exposes three diagnostics out of the box.

## Pareto-k from PSIS-LOO

Already covered in [Model comparison](@ref comparison). Recap:

```julia
l = loo(r)
@show l.pareto_k_max
```

- `< 0.7`: LOO estimate is trustworthy.
- `> 0.7`: the model is too far from at least one observation for PSIS to smooth honestly — likely misspecification.

This is the single most useful diagnostic in the package. No frequentist fitting workflow has an equivalent. On real data, when the model is structurally wrong, this lights up — quietly and reliably.

## Posterior predictive check

The [`posterior_predict`](@ref BayesBiont.posterior_predict) function samples curves from the fitted posterior:

```julia
ppc = posterior_predict(r; n_draws=500)            # n_draws × n_timepoints
ppc_mean = vec(mean(ppc; dims=1))
ppc_band = mapslices(c -> quantile(c, [0.025, 0.975]), ppc; dims=1)
```

Visual sanity check: does the band cover the data? If your observed points are systematically above or below the band over a long time window, the model isn't capturing real structure. Standard recipe:

```julia
using Plots
plot(r.times, r.observed; seriestype=:scatter, label="data")
plot!(r.times, ppc_mean; ribbon=(ppc_mean .- ppc_band[1, :], ppc_band[2, :] .- ppc_mean), label="posterior predictive")
```

PPC is more useful than a goodness-of-fit R²: it shows *where* the model fails, not just that it does.

## Chain diagnostics (Rhat, ESS)

The underlying `Chains` object is a standard MCMCChains.Chains and exposes Turing's full diagnostic suite:

```julia
using MCMCChains
summarystats(r.chains)
```

Look at:

- **Rhat** — should be close to 1.0 (typically < 1.01). > 1.05 means the chains haven't converged. Increase `n_warmup` or improve priors.
- **ESS_bulk / ESS_tail** — effective sample size. Should be > 400 for reliable quantile estimates; > 100 minimum.
- **Divergent transitions** — Turing prints warnings during sampling. >0 divergences means NUTS hit pathological geometry; usually fixed by raising `target_accept` (we default to 0.95) or using log-space hierarchy (already done internally).

## When all three diagnostics agree the fit is bad

You have three signals — Pareto-k high, posterior predictive systematically off, Rhat > 1.05. Don't trust the parameter posteriors. Either:

1. Change the model (try a different curve family; LOO `compare()` between candidates).
2. Reconsider the priors (if your priors fight the data, you'll see the posterior squeezed into prior-dominant territory).
3. Reconsider the data (preprocessing pipeline; outlier curves; replicate-vs-condition design).

## A note on calibration vs identifiability

Sometimes the posterior is *well-defined* (no divergences, Rhat ~1, Pareto-k < 0.7) but very wide. That isn't a fit failure — it's the data telling you that parameter isn't identified. A useful fit produces a tight posterior on the parameters that matter and a wide one on the parameters that don't. The Wald-CI alternative often produces a falsely-tight CI on unidentifiable parameters; BayesBiont produces an honest one.

If a posterior is much wider than you expected and the diagnostics are clean, that's a feature: your data doesn't constrain that parameter as well as the corresponding frequentist `±SE` would suggest.
