using Turing
using Distributions: Distribution, LogNormal, Normal
using Kinbiont: NLModel, ODEModel
using OrdinaryDiffEqTsit5: Tsit5
using SciMLBase: ODEProblem, remake, solve, successful_retcode, AbstractODESolution

"""
    build_turing_model(curve_func, priors_vec, sigma_prior, likelihood)

Construct a Turing `@model` closure for a single curve. `priors_vec` is an ordered
vector of `Distribution`s aligned to the curve's positional parameter vector.
"""
function build_turing_model(curve_func, priors_vec::Vector{<:Distribution},
                            sigma_prior::Distribution, likelihood::Symbol)
    if likelihood === :lognormal
        return @model function _curve_lognormal(times, y)
            p ~ arraydist(priors_vec)
            σ ~ sigma_prior
            pred = curve_func(p, times)
            # Floor against underflow when NUTS explores extreme parameter regions; the
            # shift is ~2e-16 — well below any meaningful OD — so the bias is negligible
            # while `log` and its gradient stay finite.
            for i in eachindex(y)
                y[i] ~ LogNormal(log(pred[i] + eps()), σ)
            end
        end
    elseif likelihood === :normal
        return @model function _curve_normal(times, y)
            p ~ arraydist(priors_vec)
            σ ~ sigma_prior
            pred = curve_func(p, times)
            for i in eachindex(y)
                y[i] ~ Normal(pred[i], σ)
            end
        end
    elseif likelihood === :proportional
        # Heteroscedastic noise: σ_i = σ · max(pred_i, ε). Matches Kinbiont's
        # `RE` loss semantically — relative error is constant across magnitudes.
        return @model function _curve_proportional(times, y)
            p ~ arraydist(priors_vec)
            σ ~ sigma_prior
            pred = curve_func(p, times)
            for i in eachindex(y)
                y[i] ~ Normal(pred[i], σ * (pred[i] + eps()))
            end
        end
    else
        throw(ArgumentError("unknown likelihood $(likelihood); supported: :lognormal, :normal, :proportional"))
    end
end

"""
    priors_to_vector(model, priors_nt) -> Vector{Distribution}

Order a NamedTuple of priors by the model's `param_names`. Errors on missing keys.
"""
function priors_to_vector(model::Union{NLModel, ODEModel}, priors_nt::NamedTuple)
    return [
        get(priors_nt, Symbol(name)) do
            throw(ArgumentError("missing prior for parameter `$name` of model `$(model.name)`"))
        end
        for name in model.param_names
    ]
end

"""
    _solve_ode_sum(ode_func!, p, times, y1, n_eq)

Integrate the ODE with `p` over `times` and return the per-timepoint sum of states,
or `nothing` if the solver fails or returns a partial trajectory. Callers use
`nothing` to reject the proposal with `-Inf` logprob — NUTS handles this cleanly.
"""
function _solve_ode_sum(ode_func!, p, times, y1, n_eq)
    T = eltype(p)
    u0 = vcat(convert(T, y1), zeros(T, n_eq - 1))
    tspan = (times[1], times[end])
    prob = ODEProblem(ode_func!, u0, tspan, p)
    sol = solve(prob, Tsit5(); saveat=times, abstol=1e-7, reltol=1e-5,
                save_everystep=false, verbose=false, maxiters=10_000)
    return _ode_sum(sol, length(times))
end

# Standard SciML path (Float64 and ForwardDiff Duals): solve returns an ODESolution.
function _ode_sum(sol::AbstractODESolution, n_times::Int)
    (successful_retcode(sol) && length(sol.u) == n_times) || return nothing
    return vec(sum(reduce(hcat, sol.u); dims=1))
end

# SciMLSensitivity adjoint path under ReverseDiff: solve returns a TrackedArray of
# shape (n_eq, n_times) directly — no ODESolution wrapper, no retcode field.
function _ode_sum(sol::AbstractMatrix, n_times::Int)
    size(sol, 2) == n_times || return nothing
    return vec(sum(sol; dims=1))
end

"""
    build_ode_turing_model(ode_func!, n_eq, priors_vec, sigma_prior, likelihood)

Construct a Turing `@model` for an in-place ODE system `f!(du, u, p, t)`. Observation
is the sum of all state variables at each timepoint (the standard convention for
Kinbiont's multi-compartment growth models — total cell concentration). Initial
condition puts all observed mass in the first state: `u0 = [y[1], 0, ..., 0]`.
"""
function build_ode_turing_model(ode_func!, n_eq::Int,
                                priors_vec::Vector{<:Distribution},
                                sigma_prior::Distribution, likelihood::Symbol)
    if likelihood === :lognormal
        return @model function _ode_curve_lognormal(times, y)
            p ~ arraydist(priors_vec)
            σ ~ sigma_prior

            pred = _solve_ode_sum(ode_func!, p, times, y[1], n_eq)
            if pred === nothing
                Turing.@addlogprob! -Inf
                return nothing
            end
            for i in eachindex(y)
                y[i] ~ LogNormal(log(pred[i] + eps()), σ)
            end
        end
    elseif likelihood === :normal
        return @model function _ode_curve_normal(times, y)
            p ~ arraydist(priors_vec)
            σ ~ sigma_prior

            pred = _solve_ode_sum(ode_func!, p, times, y[1], n_eq)
            if pred === nothing
                Turing.@addlogprob! -Inf
                return nothing
            end
            for i in eachindex(y)
                y[i] ~ Normal(pred[i], σ)
            end
        end
    elseif likelihood === :proportional
        return @model function _ode_curve_proportional(times, y)
            p ~ arraydist(priors_vec)
            σ ~ sigma_prior

            pred = _solve_ode_sum(ode_func!, p, times, y[1], n_eq)
            if pred === nothing
                Turing.@addlogprob! -Inf
                return nothing
            end
            for i in eachindex(y)
                y[i] ~ Normal(pred[i], σ * (pred[i] + eps()))
            end
        end
    else
        throw(ArgumentError("unknown likelihood $(likelihood); supported: :lognormal, :normal, :proportional"))
    end
end
