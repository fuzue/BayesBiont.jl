# BayesBiont.jl — Claude collaboration notes

## Mission

Bayesian companion to Kinbiont.jl. Adds calibrated UQ, hierarchical pooling across replicates, and principled model comparison. Additive ecosystem play — reuses Kinbiont's curve definitions and `GrowthData` container.

## Status

v0.1.0-DEV — currently includes most of v0.2 scope:
- ✅ Single-curve NL fits (logistic, Gompertz)
- ✅ Single-curve ODE fits (aHPM and any 2+-state ODE in Kinbiont's `MODEL_REGISTRY`)
- ✅ `:lognormal`, `:normal`, `:proportional` likelihoods
- ✅ Hierarchical pooling via `group=` kwarg (`bayesfit(data, spec; group=[...])`)
- ✅ `contrast(post, g1, g2; param)` for posterior group contrasts
- ✅ Curated `DEFAULT_PRIORS` for `NL_Gompertz`, `NL_logistic`, `aHPM`
- ✅ ReverseDiff backend (`BayesFitOptions(adbackend=:reversediff)`) — ~3× speedup on 6-curve hierarchical NL Gompertz
- ✅ ODE + ReverseDiff via `using SciMLSensitivity` — measured ~2× speedup on 6-well hierarchical aHPM (21min vs 40+ min ForwardDiff). Requires the user to add SciMLSensitivity manually because its Enzyme constraints don't always resolve with Kinbiont's transitive deps.
- 🔜 ADVI fast path — plumbing in place (`BayesFitOptions(method=:advi)`), but disabled at runtime pending Turing VI API stabilization (bijector for `arraydist`-of-LogNormal breaks during back-transform)
- 🔜 LOO/WAIC/Bayes-factor model comparison

Scope per `~/fuzue-master-plan/projects/bayesbiont.md`.

## Architectural decisions (locked in May 2026)

1. **Dependency on Kinbiont** — hard dep. Dev locally via `Pkg.develop(path="~/pinheiroTech/KinBiont.jl")`; pin registered Kinbiont in `[compat]` for release.
2. **API mirror** — reuse `Kinbiont.GrowthData` and `Kinbiont.MODEL_REGISTRY`. BayesBiont owns `BayesianModelSpec` (first-class priors). Plain `ModelSpec` accepted with auto-derived priors.
3. **Priors** — curated `DEFAULT_PRIORS` for canonical models (logistic, Gompertz, Baranyi, Richards), empirical fallback via `model.guess()` for user-defined models. Internally: log-space sampling (`Normal` on `log_param`), non-centered reparameterization for hierarchical. Users see `LogNormal` priors going in, native-scale samples coming out.
4. **Likelihood** — `likelihood=` option in `BayesFitOptions`; default `:lognormal` (matches Kinbiont's `RE` loss semantically). Defensive error on non-positive data — no silent epsilon shift.
5. **Result type** — `BayesianCurveFitResult` with lazy `posterior_predict()` and lazy `loo()`/`waic()` (storage matters for ADVI plate fits). Both `bayesfit` and `bayesian_fit` exported.
6. **NUTS defaults** — 4 chains × 1000 warmup × 1000 samples, `target_accept=0.95`, `max_treedepth=10`, init from `model.guess()` + jitter.
7. **AD backend** — `ForwardDiff` for v0.1 (single-curve, ≤6 params); `ReverseDiff` with compiled tape for v0.2 hierarchical.
8. **Multi-curve** — `bayesfit(data::GrowthData, spec)` fits each row independently in v0.1. `group=` kwarg reserved (errors until v0.2). `group_from_labels(data; pattern)` helper available v0.1.
9. **Testing** — smoke + recovery suite runs in CI (<1 min). Calibration sweep gated behind `JULIA_TEST_CALIBRATION=1`. Full coverage sweep belongs in `MisspecStudy`, not here.
10. **CI matrix** — Julia 1.11 + 1.12 (matches Kinbiont).

## Conventions

- **Internal sampling**: log-space for positive parameters; non-centered for hierarchical.
- **External display**: always native-scale via generated quantities.
- **Init**: prior medians, not Kinbiont's `guess()` heuristic — empirically NUTS warmup is more reliable from the prior, especially under `:proportional`.
- **Errors**: defensive; clear messages pointing at the fix.
- **Commits**: terse, logical, no AI attribution.

## Turing/DynamicPPL gotchas

Two real bugs surfaced during v0.2 hierarchical work, both worth knowing:

1. **`~` only works inside the `@model` macro body.** Putting `y[j] ~ Distribution` inside a regular Julia helper function compiles silently (because `~` is defined elsewhere in the namespace) but **does not add to the model's log-probability**. Symptom: all posteriors stuck at the prior. Fix: inline observation blocks directly in `@model`, no helpers.

2. **Turing's varname tracker uses the textual LHS expression.** `y_i = ys[i]; y_i[j] ~ ...` produces the varname `y_i[j]` once per loop iteration, all colliding — Turing throws "varname used multiple times in model". Fix: use `ys[i][j] ~ ...` directly so the varname includes both indices.

## Hierarchical ODE cost

Hierarchical ODE fits scale roughly as `O(n_curves * n_params^2)` per gradient eval (ForwardDiff Duals × ODE solve). Observed timings:
- 4 curves × 4 params (aHPM) × 500 warmup × 1 chain ≈ 5 min   (ForwardDiff)
- 6 curves × 4 params × 500 warmup × 2 chains ≈ 40+ min        (ForwardDiff)
- 6 curves × 4 params × 500 warmup × 2 chains ≈ **21 min**     (ReverseDiff + SciMLSensitivity)
- 6 curves × 3 params (Gompertz NL) × 300 warmup × 1 chain: 146s → **50s** (~3× from ReverseDiff alone)

## File map

```
src/
  BayesBiont.jl    module entry, exports, includes
  types.jl         BayesianModelSpec, BayesFitOptions, BayesianCurveFitResult, BayesianGrowthFitResults
  priors.jl        DEFAULT_PRIORS registry + empirical fallback
  likelihoods.jl   :normal, :lognormal Turing observation blocks
  models.jl        @model factories per Kinbiont curve
  inference.jl     NUTS dispatch, init-from-guess
  results.jl       getproperty shortcut, lazy posterior_predict
  utils.jl         group_from_labels, AD-friendly variants
  api.jl           bayesfit / bayesian_fit entry points
test/
  runtests.jl      smoke + recovery (fast)
  utils.jl         synthetic data generators
  calibration/     opt-in via JULIA_TEST_CALIBRATION=1
```

## Paired project

`MisspecStudy` — paper measuring calibration gap under structural misspecification. Depends on BayesBiont v0.2. See `~/fuzue-master-plan/projects/misspec-study.md`.

## Reference docs

- Spec: `~/fuzue-master-plan/projects/bayesbiont.md`
- Kinbiont dev source: `~/pinheiroTech/KinBiont.jl/`
