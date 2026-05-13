using Turing: NUTS, sample, MCMCSerial, MCMCThreads
using Random: MersenneTwister, AbstractRNG
using Kinbiont: NLModel, ODEModel
using MCMCChains: replacenames
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
    sampler = NUTS(opts.n_warmup, opts.target_accept;
                   max_depth=opts.max_treedepth, adtype=_ad_backend(opts.adbackend))

    init = _initial_params(guess_vec, opts, rng)

    backend = opts.n_chains == 1 ? MCMCSerial() : MCMCThreads()
    raw = sample(rng, model, sampler, backend, opts.n_samples, opts.n_chains;
                 initial_params=init, progress=false)

    return _rename_params(raw, param_names)
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
