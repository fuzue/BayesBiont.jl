using Distributions: LogNormal
using Kinbiont: AbstractGrowthModel, NLModel, ODEModel

"""
    DEFAULT_PRIORS

Curated weakly-informative LogNormal priors for canonical Kinbiont models,
grounded in typical microbial-growth literature ranges (OD scale, bacterial
growth at 25–40 °C).

| Parameter | Typical range | LogNormal centre × scale |
|---|---|---|
| `N_max` / carrying capacity | 0.1 – 3.0 OD | `LogNormal(log(1.0), 0.7)`  ≈ 95% CI `[0.25, 4]` |
| growth rate `μ` (hr⁻¹) | 0.05 – 2.0 | `LogNormal(log(0.5), 0.8)`  ≈ 95% CI `[0.10, 2.4]` |
| lag time `λ` (hr) | 0 – 24 | `LogNormal(log(5.0), 1.0)`  ≈ 95% CI `[0.7, 35]` |

These priors are appropriate for plate-reader OD data on typical bacteria. For
unusual scales (cell counts, dense cultures, slow-growing organisms) or
non-standard time units, supply explicit priors via `BayesianModelSpec(...; priors=...)`.

Models not in this registry fall back to `empirical_priors` derived from
`model.guess(data_mat)`.
"""
const DEFAULT_PRIORS = Dict{String, NamedTuple}(
    "NL_Gompertz" => (
        N_max       = LogNormal(log(1.0), 0.7),
        growth_rate = LogNormal(log(0.5), 0.8),
        lag         = LogNormal(log(5.0), 1.0),
    ),
    # Kinbiont's NL_logistic exposes params positionally [K, N_0, r] but labels
    # them ["N_max", "growth_rate", "lag"]. Priors follow Kinbiont's labels —
    # semantics is K / N_0 / r respectively.
    "NL_logistic" => (
        N_max       = LogNormal(log(1.0), 0.7),    # K
        growth_rate = LogNormal(log(0.01), 2.0),   # N_0 — very wide; weakly identified
        lag         = LogNormal(log(0.5), 0.8),    # r
    ),
    # aHPM (adjusted heterogeneous population model) — 2-state ODE.
    # States: u[1] = dormant pool, u[2] = active growth.
    # Params: gr, exit_lag_rate, N_max, shape.
    "aHPM" => (
        gr            = LogNormal(log(0.5), 0.8),  # specific growth rate hr⁻¹
        exit_lag_rate = LogNormal(log(0.2), 1.0),  # lag-exit rate hr⁻¹
        N_max         = LogNormal(log(1.0), 0.7),  # carrying capacity (OD)
        shape         = LogNormal(log(1.0), 0.5),  # logistic shape (1 = standard)
    ),
)

"""
    default_priors(model, data_mat) -> NamedTuple

Curated biology-grounded priors when `model.name` is in [`DEFAULT_PRIORS`];
otherwise empirical `LogNormal(log(model.guess(data_mat)), 1.0)` per parameter.
"""
function default_priors(model::NLModel, data_mat::AbstractMatrix)
    haskey(DEFAULT_PRIORS, model.name) && return DEFAULT_PRIORS[model.name]
    return empirical_priors(model, data_mat)
end

function default_priors(model::ODEModel, data_mat::AbstractMatrix)
    haskey(DEFAULT_PRIORS, model.name) && return DEFAULT_PRIORS[model.name]
    model.guess === nothing && throw(ArgumentError(
        "ODE model `$(model.name)` has no guess function and no entry in DEFAULT_PRIORS; " *
        "pass explicit priors via BayesianModelSpec(...; priors=...)"))
    return empirical_priors_ode(model, data_mat)
end

function empirical_priors_ode(model::ODEModel, data_mat::AbstractMatrix)
    guess_vec = model.guess(data_mat)
    return NamedTuple(
        Symbol(name) => LogNormal(log(max(g, eps())), 1.0)
        for (name, g) in zip(model.param_names, guess_vec)
    )
end

"""
    empirical_priors(model, data_mat) -> NamedTuple

Weakly-informative LogNormal priors derived from `model.guess(data_mat)`. Each
parameter gets `LogNormal(log(guess_p), 1.0)` — a 95% CI spanning ×7 either way.
"""
function empirical_priors(model::NLModel, data_mat::AbstractMatrix)
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
