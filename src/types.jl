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

- `likelihood::Symbol = :lognormal`: `:lognormal` (multiplicative noise, requires positive
  data) or `:normal` (additive).
- `n_chains::Int = 4`, `n_warmup::Int = 1000`, `n_samples::Int = 1000`
- `target_accept::Float64 = 0.95`: NUTS dual-averaging target.
- `max_treedepth::Int = 10`
- `jitter::Float64 = 0.1`: log-space jitter around `model.guess()` for per-chain init.
- `rng_seed::Union{Nothing, Int} = nothing`: deterministic seed when set.
- `adbackend::Symbol = :forwarddiff`: reserved for v0.2 (`:reversediff` planned).
"""
Base.@kwdef struct BayesFitOptions
    likelihood::Symbol               = :lognormal
    n_chains::Int                    = 4
    n_warmup::Int                    = 1000
    n_samples::Int                   = 1000
    target_accept::Float64           = 0.95
    max_treedepth::Int               = 10
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
