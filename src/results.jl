using MCMCChains: Chains, namesingroup

"""
    Base.getproperty(r::BayesianCurveFitResult, name::Symbol)

Field access shortcut: declared struct fields fall through; otherwise treat `name` as
a parameter symbol in the underlying chain and return a flat sample vector across
chains and iterations.
"""
function Base.getproperty(r::BayesianCurveFitResult, name::Symbol)
    if name in fieldnames(BayesianCurveFitResult)
        return getfield(r, name)
    end
    chains = getfield(r, :chains)
    chain_params = names(chains, :parameters)
    sym_name = name
    if sym_name in chain_params
        arr = chains[sym_name].data
        return vec(arr)
    end
    throw(ArgumentError("`$name` is neither a field of BayesianCurveFitResult nor a sampled parameter"))
end

Base.propertynames(r::BayesianCurveFitResult) =
    (fieldnames(BayesianCurveFitResult)..., Tuple(names(getfield(r, :chains), :parameters))...)

"""
    posterior_predict(result::BayesianCurveFitResult; n_draws=200) -> Matrix{Float64}

Posterior predictive curve samples on `result.times`. Returns an `n_draws × n_timepoints`
matrix; row `i` is the deterministic curve evaluated at posterior draw `i`. Use
`vec(mean(out, dims=1))` for the posterior mean curve and
`mapslices(c -> quantile(c, [0.025, 0.975]), out; dims=1)` for pointwise 95% bands.

Lazy by design — not cached on `result`; recompute when you need different `n_draws`.
"""
function posterior_predict(r::BayesianCurveFitResult; n_draws::Int=200)
    chain_params = names(r.chains, :parameters)
    p_names = filter(!=(:σ), chain_params)
    param_arrays = [vec(r.chains[p].data) for p in p_names]
    total = length(param_arrays[1])
    n_draws = min(n_draws, total)
    idx = round.(Int, range(1, total; length=n_draws))

    n_t = length(r.times)
    out = Matrix{Float64}(undef, n_draws, n_t)
    for (j, k) in enumerate(idx)
        p_vec = [arr[k] for arr in param_arrays]
        out[j, :] = r.model.func(p_vec, r.times)
    end
    return out
end
