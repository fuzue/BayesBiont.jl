using Kinbiont: GrowthData, AbstractGrowthModel, NLModel
using Distributions

"""
    bayesfit(data::GrowthData, spec::BayesianModelSpec[, opts::BayesFitOptions]; group=nothing)

Fit each curve in `data` Bayesianly under `spec`. Returns `BayesianGrowthFitResults`.

v0.1 fits each curve independently. The `group=` kwarg is reserved for hierarchical
pooling in v0.2 and errors today.

For multi-model `spec` (more than one candidate model), v0.1 fits only the first model
per curve. Model comparison via LOO/WAIC lands in v0.3.

# Example

```julia
using BayesBiont, Kinbiont, Statistics

times = collect(0.0:0.25:24.0)
curve = 1.0 .* exp.(-exp.(-0.4 .* (times .- 5.0)))
data  = GrowthData(reshape(curve, 1, :), times, ["well1"])

spec  = BayesianModelSpec([MODEL_REGISTRY["NL_Gompertz"]])
post  = bayesfit(data, spec, BayesFitOptions(n_chains=2, n_warmup=400, n_samples=400))

r = post[1]
mean(r.growth_rate), quantile(r.growth_rate, [0.025, 0.975])
```
"""
function bayesfit(data::GrowthData, spec::BayesianModelSpec,
                  opts::BayesFitOptions = BayesFitOptions(); group=nothing)
    isempty(spec.models) && throw(ArgumentError("BayesianModelSpec has no models"))
    model = first(spec.models)        # single model per spec for now

    if group !== nothing
        return _fit_hierarchical(data, spec, opts, model, group)
    end

    results = map(axes(data.curves, 1)) do i
        y = collect(data.curves[i, :])
        check_likelihood_data!(opts.likelihood, y)
        _fit_one(model, data.times, y, data.labels[i], spec, opts)
    end

    return BayesianGrowthFitResults(data, results)
end

function _fit_hierarchical(data, spec, opts, model, group)
    model isa ODEModel && _check_ode_adbackend(opts.adbackend)
    length(group) == size(data.curves, 1) ||
        throw(ArgumentError("`group` length $(length(group)) ≠ number of curves $(size(data.curves, 1))"))

    group_labels = String.(group)
    groups       = sort(unique(group_labels))
    group_idx    = [findfirst(==(g), groups) for g in group_labels]

    ys = [collect(data.curves[i, :]) for i in axes(data.curves, 1)]
    foreach(y -> check_likelihood_data!(opts.likelihood, y), ys)

    data_mat   = Matrix(transpose(hcat(data.times, ys[1])))  # representative for priors
    priors_nt  = _resolve_priors(spec, model, data_mat)
    priors_vec = priors_to_vector(model, priors_nt)

    prior_tau = 0.5    # HalfNormal scale on log-space spread; v0.2.x: expose in BayesFitOptions
    turing_model = build_hierarchical_turing_model(model, priors_vec, spec.sigma_prior,
                                                   prior_tau, group_idx, opts.likelihood)

    rng = opts.rng_seed === nothing ? Random.MersenneTwister() : Random.MersenneTwister(opts.rng_seed)
    sampler = Turing.NUTS(opts.n_warmup, opts.target_accept;
                          max_depth=opts.max_treedepth, adtype=_ad_backend(opts.adbackend))
    backend = opts.n_chains == 1 ? Turing.MCMCSerial() : Turing.MCMCThreads()

    n_params = length(priors_vec)
    n_curves = length(ys)
    n_groups = length(groups)
    log_means = [_lognormal_log_mean(p) for p in priors_vec]
    init = [(
        μ_pop = repeat(log_means', n_groups),
        τ     = fill(0.1, n_params),
        z     = zeros(n_curves, n_params),
        σ_obs = fill(0.1, n_curves),
    ) for _ in 1:opts.n_chains]

    raw = Turing.sample(rng, turing_model(data.times, ys), sampler, backend,
                        opts.n_samples, opts.n_chains;
                        initial_params=init, progress=false)
    chains = _rename_hier_params(raw, model, group_labels, groups)

    return HierarchicalBayesianFitResults(data, model, group_labels, groups, chains,
                                          data.times, ys)
end

"""
    bayesian_fit(args...; kwargs...)

Snake-case alias for [`bayesfit`](@ref). Provided for Kinbiont users who prefer
`kinbiont_fit`-style naming.
"""
const bayesian_fit = bayesfit

function _fit_one(model::NLModel, times::Vector{Float64}, y::Vector{Float64},
                  label::String, spec::BayesianModelSpec, opts::BayesFitOptions)
    data_mat = Matrix(transpose(hcat(times, y)))
    priors_nt = _resolve_priors(spec, model, data_mat)
    priors_vec = priors_to_vector(model, priors_nt)
    init_vec = _init_from_priors(priors_vec, model, data_mat)
    turing_model = build_turing_model(model.func, priors_vec, spec.sigma_prior, opts.likelihood)
    chains = fit_single_curve(turing_model, model.param_names, times, y, init_vec, opts)
    return BayesianCurveFitResult(label, model, chains, times, y)
end

function _fit_one(model::ODEModel, times::Vector{Float64}, y::Vector{Float64},
                  label::String, spec::BayesianModelSpec, opts::BayesFitOptions)
    _check_ode_adbackend(opts.adbackend)
    data_mat = Matrix(transpose(hcat(times, y)))
    priors_nt = _resolve_priors(spec, model, data_mat)
    priors_vec = priors_to_vector(model, priors_nt)
    init_vec = _init_from_priors(priors_vec, model, data_mat)
    turing_model = build_ode_turing_model(model.func, model.n_eq, priors_vec,
                                          spec.sigma_prior, opts.likelihood)
    chains = fit_single_curve(turing_model, model.param_names, times, y, init_vec, opts)
    return BayesianCurveFitResult(label, model, chains, times, y)
end

function _check_ode_adbackend(adbackend::Symbol)
    adbackend === :reversediff && throw(ArgumentError(
        "ODE models with `adbackend=:reversediff` require SciMLSensitivity.jl to be " *
        "loaded in your session (`using SciMLSensitivity`). It is not a hard dependency " *
        "of BayesBiont because its Enzyme version constraints conflict with Kinbiont's " *
        "Optimization.jl chain. Either: (a) `Pkg.add(\"SciMLSensitivity\")` in your " *
        "environment and `using SciMLSensitivity` before calling `bayesfit`, or " *
        "(b) keep `adbackend=:forwarddiff` (default) for ODE fits."))
    return nothing
end

# Initial values prefer the prior median (biology-informed centre) over Kinbiont's
# MLE-oriented `guess()` heuristic — empirically NUTS warmup is more reliable from
# the prior, especially under heteroscedastic / proportional likelihoods where bad
# init traps the sampler. Fall back to `guess()` only when the priors are uniform-ish
# (no curated entry and no user override).
function _init_from_priors(priors_vec, model, data_mat)
    return [_prior_init_value(p) for p in priors_vec]
end

_prior_init_value(p::Distributions.LogNormal) = exp(p.μ)
_prior_init_value(p) = Distributions.median(p)

_fit_one(model::AbstractGrowthModel, ::Vector{Float64}, ::Vector{Float64},
         ::String, ::BayesianModelSpec, ::BayesFitOptions) =
    throw(ArgumentError("BayesBiont does not yet support $(typeof(model))"))

function _resolve_priors(spec::BayesianModelSpec, model, data_mat)
    spec.priors === nothing && return default_priors(model, data_mat)
    return first(spec.priors)::NamedTuple
end
