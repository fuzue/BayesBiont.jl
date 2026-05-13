using Turing: NUTS, ADVI, sample, vi, MCMCSerial, MCMCThreads
using Random: MersenneTwister, AbstractRNG, default_rng
using Kinbiont: NLModel, ODEModel
using MCMCChains: Chains, replacenames
using Distributions: mean, Distribution
using ADTypes: AutoForwardDiff, AutoReverseDiff

"""
    _ad_backend(adbackend::Symbol)

Map a user-facing symbol to an ADTypes AD backend. Supports `:forwarddiff` (default,
robust, fast for ≤6 parameters) and `:reversediff` (compiled tape, much faster for
hierarchical models with 50+ latent variables but slightly heavier first call).
"""
function _ad_backend(adbackend::Symbol)
    if adbackend === :forwarddiff
        return AutoForwardDiff()
    elseif adbackend === :reversediff
        return AutoReverseDiff(; compile=true)
    else
        throw(ArgumentError("unknown adbackend $(adbackend); supported: :forwarddiff, :reversediff"))
    end
end

"""
    fit_single_curve(curve_func, param_names, times, y, priors_vec, sigma_prior, guess_vec, opts)

Run NUTS on a single curve and return a `Chains` with parameters renamed from the
internal `p[i]` scheme to the model's `param_names`.

Initial values per chain are `guess_vec` perturbed in log-space by `opts.jitter * randn()`.
"""
function fit_single_curve(turing_model, param_names::Vector{String},
                          times::Vector{Float64}, y::Vector{Float64},
                          guess_vec::Vector{Float64}, opts)
    rng = opts.rng_seed === nothing ? MersenneTwister() : MersenneTwister(opts.rng_seed)
    model = turing_model(times, y)

    if opts.method === :advi
        return _fit_advi(rng, model, param_names, opts)
    elseif opts.method === :nuts
        return _fit_nuts(rng, model, param_names, guess_vec, opts)
    else
        throw(ArgumentError("unknown method $(opts.method); supported: :nuts, :advi"))
    end
end

function _fit_nuts(rng, model, param_names, guess_vec, opts)
    sampler = NUTS(opts.n_warmup, opts.target_accept;
                   max_depth=opts.max_treedepth, adtype=_ad_backend(opts.adbackend))
    init = _initial_params(guess_vec, opts, rng)
    backend = opts.n_chains == 1 ? MCMCSerial() : MCMCThreads()
    raw = sample(rng, model, sampler, backend, opts.n_samples, opts.n_chains;
                 initial_params=init, progress=false)
    return _rename_params(raw, param_names)
end

# ADVI: mean-field variational inference. Fits a Gaussian in the unconstrained
# space, transforms back, and draws `n_samples` from the approximate posterior.
# Wrapping samples in a Chains keeps the downstream API (getproperty, posterior_predict)
# unchanged from the NUTS path.
function _fit_advi(rng, model, param_names, opts)
    # EXPERIMENTAL — disabled pending Turing VI API stabilization.
    # The ELBO optimization runs to convergence, but Turing's bijector for
    # `arraydist` of heterogeneous LogNormal priors doesn't propagate correctly,
    # causing back-transform errors during sample extraction. Re-enable once
    # AdvancedVI exposes a stable per-element bijector API.
    throw(ArgumentError(
        "method=:advi is experimental and currently disabled in BayesBiont " *
        "(Turing's bijector for arraydist-of-LogNormal under VI is unstable). " *
        "Use method=:nuts (default) for now. Track Turing.jl VI API progress " *
        "at https://github.com/TuringLang/Turing.jl/issues"))
end

function _initial_params(guess_vec::Vector{Float64}, opts, rng::AbstractRNG)
    n = length(guess_vec)
    log_guess = log.(max.(guess_vec, eps()))
    chains = opts.n_chains
    return [
        (p = exp.(log_guess .+ opts.jitter .* randn(rng, n)), σ = 0.1)
        for _ in 1:chains
    ]
end

function _rename_params(chains, param_names::Vector{String})
    renames = ["p[$i]" => param_names[i] for i in eachindex(param_names)]
    return replacenames(chains, renames...)
end
