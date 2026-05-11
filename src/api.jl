using Kinbiont: GrowthData, AbstractGrowthModel, NLModel

"""
    bayesfit(data::GrowthData, spec::BayesianModelSpec[, opts::BayesFitOptions]; group=nothing)

Fit each curve in `data` Bayesianly under `spec`. Returns `BayesianGrowthFitResults`.

v0.1 fits each curve independently. The `group=` kwarg is reserved for hierarchical
pooling in v0.2 and errors today.

For multi-model `spec` (more than one candidate model), v0.1 fits only the first model
per curve. Model comparison via LOO/WAIC lands in v0.3.
"""
function bayesfit(data::GrowthData, spec::BayesianModelSpec,
                  opts::BayesFitOptions = BayesFitOptions(); group=nothing)
    group === nothing || throw(ArgumentError(
        "hierarchical pooling (`group=`) lands in v0.2; v0.1 fits curves independently"))

    isempty(spec.models) && throw(ArgumentError("BayesianModelSpec has no models"))
    model = first(spec.models)        # v0.1: single model per spec

    results = map(axes(data.curves, 1)) do i
        y = collect(data.curves[i, :])
        check_likelihood_data!(opts.likelihood, y)
        _fit_one(model, data.times, y, data.labels[i], spec, opts)
    end

    return BayesianGrowthFitResults(data, results)
end

const bayesian_fit = bayesfit

function _fit_one(model::NLModel, times::Vector{Float64}, y::Vector{Float64},
                  label::String, spec::BayesianModelSpec, opts::BayesFitOptions)
    data_mat = Matrix(transpose(hcat(times, y)))

    priors_nt = if spec.priors === nothing
        default_priors(model, data_mat)
    else
        first(spec.priors)::NamedTuple
    end
    priors_vec = priors_to_vector(model, priors_nt)

    guess_vec = model.guess === nothing ?
        [mean(p) for p in priors_vec] :
        model.guess(data_mat)

    chains = fit_single_curve(model.func, model.param_names, times, y,
                              priors_vec, spec.sigma_prior, guess_vec, opts)
    return BayesianCurveFitResult(label, model, chains, times, y)
end

_fit_one(model::AbstractGrowthModel, ::Vector{Float64}, ::Vector{Float64},
         ::String, ::BayesianModelSpec, ::BayesFitOptions) =
    throw(ArgumentError("v0.1 supports NLModel only; got $(typeof(model))"))
