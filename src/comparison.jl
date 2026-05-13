using Distributions: LogNormal, Normal, logpdf
using PSIS: psis
using Statistics: mean, var, std
using Kinbiont: NLModel, ODEModel

# Numerically stable log-sum-exp; inlined to avoid pulling in LogExpFunctions for one helper.
function _logsumexp(xs)
    m = maximum(xs)
    isfinite(m) || return m
    return m + log(sum(exp(x - m) for x in xs))
end

"""
    pointwise_loglik(r::BayesianCurveFitResult) -> Matrix{Float64}

Per-(sample, observation) log-likelihood matrix, shape `(n_samples, n_obs)`.
Recomputes the model at each posterior draw and applies the same observation
density used during fitting (`r.likelihood`).
"""
function pointwise_loglik(r::BayesianCurveFitResult)
    p_names = filter(!=(:σ), names(r.chains, :parameters))
    param_arrays = [vec(r.chains[p].data) for p in p_names]
    σ_array = vec(r.chains[:σ].data)
    n_samples = length(σ_array)
    n_obs = length(r.observed)

    ll = Matrix{Float64}(undef, n_samples, n_obs)
    for s in 1:n_samples
        p_vec = [arr[s] for arr in param_arrays]
        pred  = _evaluate_curve(r.model, p_vec, r.times, r.observed)
        σ     = σ_array[s]
        for i in 1:n_obs
            ll[s, i] = _obs_logpdf(r.likelihood, pred[i], σ, r.observed[i])
        end
    end
    return ll
end

_obs_logpdf(::Val{:lognormal}, pred, σ, y) =
    logpdf(LogNormal(log(pred + eps()), σ), y)
_obs_logpdf(::Val{:normal}, pred, σ, y) =
    logpdf(Normal(pred, σ), y)
_obs_logpdf(::Val{:proportional}, pred, σ, y) =
    logpdf(Normal(pred, σ * (pred + eps())), y)
_obs_logpdf(likelihood::Symbol, pred, σ, y) = _obs_logpdf(Val(likelihood), pred, σ, y)

"""
    waic(r::BayesianCurveFitResult) -> (elpd, p_eff, n_obs)

Watanabe–Akaike Information Criterion (WAIC) estimate of out-of-sample predictive
performance. Returns:

- `elpd`: expected log pointwise predictive density (higher = better fit)
- `p_eff`: effective number of parameters (penalty)
- `n_obs`: number of observations used

WAIC tends to be unreliable when individual observations strongly influence the
posterior — prefer `loo` (PSIS-LOO) for robust model comparison.
"""
function waic(r::BayesianCurveFitResult)
    ll = pointwise_loglik(r)
    n_obs = size(ll, 2)
    lpd = sum(log(mean(exp.(ll[:, i]))) for i in 1:n_obs)
    p_eff = sum(var(ll[:, i]) for i in 1:n_obs)
    return (elpd = lpd - p_eff, p_eff = p_eff, n_obs = n_obs)
end

"""
    loo(r::BayesianCurveFitResult) -> (elpd, se, n_obs, pareto_k_max, elpd_pointwise)

Leave-one-out cross-validation expected log predictive density via Pareto-smoothed
importance sampling (PSIS-LOO). Returns:

- `elpd`: PSIS-LOO ELPD estimate (higher = better fit)
- `se`: standard error of the ELPD estimate
- `n_obs`: number of observations
- `pareto_k_max`: maximum Pareto-k diagnostic. Values > 0.7 indicate the
  importance-sampling approximation is unreliable for some observation;
  > 1.0 means PSIS-LOO cannot be trusted on this data.
- `elpd_pointwise`: per-observation ELPD contributions (useful for `compare`).
"""
function loo(r::BayesianCurveFitResult)
    ll = pointwise_loglik(r)           # (n_samples, n_obs)
    return _loo_from_ll(ll)
end

# Compute PSIS-LOO ELPD from a (n_samples, n_obs) log-likelihood matrix.
function _loo_from_ll(ll::AbstractMatrix)
    n_samples, n_obs = size(ll)
    # Importance ratios for leave-i-out: log r[s,i] = -log_lik[s,i]
    log_w_raw = .-ll
    psis_result = psis(log_w_raw)
    log_w = psis_result.log_weights     # smoothed (n_samples, n_obs)

    elpd_i = Vector{Float64}(undef, n_obs)
    for i in 1:n_obs
        elpd_i[i] = _logsumexp(view(ll, :, i) .+ view(log_w, :, i)) -
                    _logsumexp(view(log_w, :, i))
    end
    return (
        elpd = sum(elpd_i),
        se   = sqrt(n_obs) * std(elpd_i),
        n_obs = n_obs,
        pareto_k_max = maximum(psis_result.pareto_shape),
        elpd_pointwise = elpd_i,
    )
end

"""
    compare(r1, r2) -> (elpd_diff, se_diff, favours)

Compare two `BayesianCurveFitResult`s by PSIS-LOO ELPD. Returns the difference
`elpd(r1) − elpd(r2)`, its standard error, and a string naming which model is
favoured. A difference of more than ~2 SEs is conventionally treated as a
meaningful preference; smaller differences mean the data don't strongly
distinguish the two models.
"""
function compare(r1::BayesianCurveFitResult, r2::BayesianCurveFitResult)
    ll1 = pointwise_loglik(r1)
    ll2 = pointwise_loglik(r2)
    size(ll1, 2) == size(ll2, 2) ||
        throw(ArgumentError("models compared on different observation counts " *
                            "($(size(ll1,2)) vs $(size(ll2,2)))"))

    elpd_i_1 = _loo_from_ll(ll1).elpd_pointwise
    elpd_i_2 = _loo_from_ll(ll2).elpd_pointwise

    diff_i = elpd_i_1 .- elpd_i_2
    elpd_diff = sum(diff_i)
    se_diff = sqrt(length(diff_i)) * std(diff_i)

    favours = if elpd_diff > 2 * se_diff
        "$(r1.model.name) (>2 SE)"
    elseif elpd_diff < -2 * se_diff
        "$(r2.model.name) (>2 SE)"
    else
        "no strong preference (|Δ| < 2·SE)"
    end

    return (elpd_diff = elpd_diff, se_diff = se_diff, favours = favours)
end
