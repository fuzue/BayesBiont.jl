# [Hierarchical pooling](@id hierarchical)

```@contents
Pages = ["index.md"]
Depth = 2
```

When you have multiple replicates of the same biological condition (multiple wells, multiple days, multiple plates), the right thing to do is **partial pooling** — not "average then fit" and not "fit independently then average". BayesBiont's `group=` keyword does this.

## The basic call

```julia
data  = GrowthData(curve_matrix, times, labels)
spec  = BayesianModelSpec([MODEL_REGISTRY["NL_Gompertz"]])
group = ["WT", "WT", "WT", "mut", "mut", "mut"]      # one label per curve in `data`

post = bayesfit(data, spec; group=group)             # HierarchicalBayesianFitResults
```

The model fitted has, per parameter:

1. **Population-level mean** `μ_pop[g, k]` on log-scale, per group `g`, parameter `k`.
2. **Population-level spread** `τ[k]`, shared across groups.
3. **Per-curve standardised offset** `z[i, k] ~ Normal(0, 1)`.
4. **Per-curve parameter** `p[i, k] = exp(μ_pop[group(i), k] + τ[k] * z[i, k])`.

This is the standard non-centered hierarchical reparameterisation, which avoids the funnel geometry that breaks centered models under NUTS.

## Looking at the results

`post` is a `HierarchicalBayesianFitResults`. Its REPL show prints group-level summaries:

```
HierarchicalBayesianFitResults — 6 curves over 2 group(s):
  [WT]  3 curve(s): w1, w2, w3
  [mut] 3 curve(s): m1, m2, m3
  Population means (native scale, per group):
    WT.growth_rate    mean=0.401  95% CI=[0.376, 0.426]
    WT.N_max          mean=0.997  95% CI=[0.962, 1.034]
    ...
```

Access the underlying `Chains` directly with `post.chains` — the population means are named `μ_pop_<group>_<param>` (still in log-space; exponentiate to get native scale).

## Contrasts

The [`contrast`](@ref BayesBiont.contrast) function computes posterior probability statements about inter-group differences:

```julia
diff = contrast(post, "WT", "mut"; param=:growth_rate)
# Vector{Float64} — posterior samples of (exp(μ_pop[WT, gr]) - exp(μ_pop[mut, gr]))

println("P(WT faster than mut) = ", mean(diff .> 0))
println("Δgrowth_rate 95% CI = ", quantile(diff, [0.025, 0.975]))
```

This is the answer to "is the wild type really faster?" expressed as a probability statement, not a t-test on point estimates.

## Verified speed

Measured on real LG175 plate-reader data (6 wells × aHPM ODE × 2 chains × 500 warmup):

- ForwardDiff: 40+ min
- ReverseDiff + SciMLSensitivity: **21 min**
- Same posterior to plotting precision.

On NL hierarchical (6-curve Gompertz, 1 chain × 300 warmup + 300 samples):

- ForwardDiff: 146 s
- ReverseDiff: **50 s**

Set `BayesFitOptions(adbackend=:reversediff)` (NL works out of the box; ODE additionally needs `using SciMLSensitivity`).

## Public API

- [`HierarchicalBayesianFitResults`](@ref BayesBiont.HierarchicalBayesianFitResults)
- [`contrast`](@ref BayesBiont.contrast)
- [`group_from_labels`](@ref BayesBiont.group_from_labels)
