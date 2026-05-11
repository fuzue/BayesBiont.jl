using Distributions: LogNormal
using Kinbiont: AbstractGrowthModel, NLModel

"""
    default_priors(model, data_mat) -> NamedTuple

Empirical fallback: derive weakly-informative `LogNormal` priors from
`model.guess(data_mat)`. Each parameter `p` gets `LogNormal(log(guess_p), 1.0)` — a
95% CI spanning roughly ×7 around the guess, which is wide enough to be honest
and tight enough for NUTS to mix.

A curated `DEFAULT_PRIORS` registry for canonical models (logistic, Gompertz,
Baranyi, Richards) will override this in v0.2/v0.3.
"""
function default_priors(model::NLModel, data_mat::AbstractMatrix)
    model.guess === nothing &&
        throw(ArgumentError("model `$(model.name)` has no guess function; pass explicit priors"))
    guess_vec = model.guess(data_mat)
    return NamedTuple(
        Symbol(name) => LogNormal(log(max(g, eps())), 1.0)
        for (name, g) in zip(model.param_names, guess_vec)
    )
end

default_priors(model::AbstractGrowthModel, ::AbstractMatrix) =
    throw(ArgumentError("automatic priors not yet supported for $(typeof(model)); pass explicit priors"))
