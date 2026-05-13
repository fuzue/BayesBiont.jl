using Kinbiont: AbstractGrowthModel
using Distributions: Distribution, Exponential

"""
    BayesianModelSpec(models; priors=nothing, sigma_prior=Exponential(0.1))

Specifies which Kinbiont models to fit Bayesianly and (optionally) their priors.

- `models::Vector{<:AbstractGrowthModel}`: candidate models, typically from `Kinbiont.MODEL_REGISTRY`.
- `priors::Union{Nothing, Vector}`: one prior `NamedTuple` per model, keyed by parameter
  name (`Symbol`). When `nothing`, BayesBiont derives weakly-informative `LogNormal`
  priors from `model.guess(data_mat)` at fit time.
- `sigma_prior::Distribution`: prior on the observation-noise scale `σ`.
"""
struct BayesianModelSpec
    models::Vector{<:AbstractGrowthModel}
    priors::Union{Nothing, Vector}
    sigma_prior::Distribution
end

BayesianModelSpec(models; priors=nothing, sigma_prior=Exponential(0.1)) =
    BayesianModelSpec(collect(models), priors, sigma_prior)

"""
    BayesFitOptions(; kwargs...)

Configuration for Bayesian fitting.

- `method::Symbol = :nuts`: inference algorithm. `:nuts` for calibrated posteriors,
  `:advi` for fast variational approximation (mean-field; underestimates uncertainty
  but typically 10–100× faster — useful for plate-scale screening, follow up with NUTS
  on wells of interest).
- `likelihood::Symbol = :lognormal`: `:lognormal`, `:normal`, or `:proportional`.
- `n_chains::Int = 4`, `n_warmup::Int = 1000`, `n_samples::Int = 1000` — NUTS only.
- `target_accept::Float64 = 0.95`: NUTS dual-averaging target.
- `max_treedepth::Int = 10`: NUTS only.
- `advi_n_iters::Int = 5000`: ADVI optimisation iterations.
- `advi_samples_per_step::Int = 10`: ADVI ELBO Monte Carlo samples per gradient step.
- `jitter::Float64 = 0.1`: log-space jitter around `model.guess()` for per-chain init.
- `rng_seed::Union{Nothing, Int} = nothing`: deterministic seed when set.
- `adbackend::Symbol = :forwarddiff`: `:forwarddiff` (default) or `:reversediff`.
"""
Base.@kwdef struct BayesFitOptions
    method::Symbol                   = :nuts
    likelihood::Symbol               = :lognormal
    n_chains::Int                    = 4
    n_warmup::Int                    = 1000
    n_samples::Int                   = 1000
    target_accept::Float64           = 0.95
    max_treedepth::Int               = 10
    advi_n_iters::Int                = 5000
    advi_samples_per_step::Int       = 10
    jitter::Float64                  = 0.1
    rng_seed::Union{Nothing, Int}    = nothing
    adbackend::Symbol                = :forwarddiff
end

"""
    BayesianCurveFitResult(label, model, chains, times, observed)

Bayesian fit result for a single growth curve. Field access shortcut: `result.param_name`
returns a flat sample vector for any parameter in the underlying `Chains`.
"""
struct BayesianCurveFitResult
    label::String
    model::AbstractGrowthModel
    chains::Any                       # MCMCChains.Chains (kept untyped to avoid heavy load order)
    times::Vector{Float64}
    observed::Vector{Float64}
end

"""
    BayesianGrowthFitResults(data, results)

Container for per-curve Bayesian fits. Iterable; indexable by integer; carries the
input `GrowthData` for downstream reference.
"""
struct BayesianGrowthFitResults
    data::Any                         # Kinbiont.GrowthData (kept untyped for load order)
    results::Vector{BayesianCurveFitResult}
end

Base.length(r::BayesianGrowthFitResults)            = length(r.results)
Base.iterate(r::BayesianGrowthFitResults, args...)  = iterate(r.results, args...)
Base.getindex(r::BayesianGrowthFitResults, i::Int)  = r.results[i]
