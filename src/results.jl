using MCMCChains: Chains, namesingroup
using Statistics: mean, quantile
using Kinbiont: NLModel, ODEModel
using OrdinaryDiffEqTsit5: Tsit5
using SciMLBase: ODEProblem, solve

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
        out[j, :] = _evaluate_curve(r.model, p_vec, r.times, r.observed)
    end
    return out
end

_evaluate_curve(model::NLModel, p, times, _) = model.func(p, times)

function _evaluate_curve(model::ODEModel, p, times, observed)
    u0 = vcat(observed[1], zeros(model.n_eq - 1))
    prob = ODEProblem(model.func, u0, (times[1], times[end]), p)
    sol = solve(prob, Tsit5(); saveat=times, abstol=1e-7, reltol=1e-5,
                save_everystep=false)
    return vec(sum(reduce(hcat, sol.u); dims=1))
end

function Base.show(io::IO, ::MIME"text/plain", r::BayesianCurveFitResult)
    n_samples = length(vec(r.chains[Symbol(first(r.model.param_names))].data))
    println(io, "BayesianCurveFitResult(label=\"$(r.label)\", model=\"$(r.model.name)\")")
    println(io, "  $(length(r.times)) timepoints, $n_samples total posterior samples")
    println(io, "  Parameters:")
    for name in r.model.param_names
        samples = vec(r.chains[Symbol(name)].data)
        m = mean(samples)
        lo, hi = quantile(samples, [0.025, 0.975])
        println(io, "    $(rpad(name, 14)) mean=$(round(m; sigdigits=4))  " *
                    "95% CI=[$(round(lo; sigdigits=4)), $(round(hi; sigdigits=4))]")
    end
    σ_samples = vec(r.chains[:σ].data)
    print(io, "    $(rpad("σ", 14)) mean=$(round(mean(σ_samples); sigdigits=4))")
end

function Base.show(io::IO, ::MIME"text/plain", r::BayesianGrowthFitResults)
    n = length(r.results)
    println(io, "BayesianGrowthFitResults with $n curve$(n == 1 ? "" : "s"):")
    for (i, cr) in enumerate(r.results)
        print(io, "  [$i] \"$(cr.label)\" → $(cr.model.name)")
        i < n && println(io)
    end
end
