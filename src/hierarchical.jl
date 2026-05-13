using Turing
using Distributions: Normal, LogNormal, truncated
using Kinbiont: NLModel, ODEModel, AbstractGrowthModel
using MCMCChains: Chains, replacenames

"""
    HierarchicalBayesianFitResults

Result of a hierarchical `bayesfit` with `group=`. Holds:
- `data`         — original `GrowthData`
- `model`        — Kinbiont model fitted
- `group_labels` — group string per row of `data`
- `groups`       — ordered unique group names
- `chains`       — single big `Chains` with all parameters
- `times`, `observed_per_curve` — input data echoed for downstream evaluation
"""
struct HierarchicalBayesianFitResults
    data::Any
    model::AbstractGrowthModel
    group_labels::Vector{String}
    groups::Vector{String}
    chains::Any
    times::Vector{Float64}
    observed_per_curve::Vector{Vector{Float64}}
end

Base.length(r::HierarchicalBayesianFitResults) = length(r.observed_per_curve)

function Base.show(io::IO, ::MIME"text/plain", r::HierarchicalBayesianFitResults)
    println(io, "HierarchicalBayesianFitResults — $(length(r)) curves over $(length(r.groups)) group(s):")
    for g in r.groups
        idxs = findall(==(g), r.group_labels)
        println(io, "  [$g] $(length(idxs)) curve(s): $(join(r.data.labels[idxs], ", "))")
    end
    println(io, "  Population means (native scale, per group):")
    for g in r.groups, name in r.model.param_names
        sym = Symbol("μ_pop_", g, "_", name)
        samples = exp.(vec(r.chains[sym].data))
        lo, hi  = quantile(samples, [0.025, 0.975])
        print(io, "    $(rpad("$g.$name", 22))  mean=$(round(mean(samples); sigdigits=4))  " *
                  "95% CI=[$(round(lo; sigdigits=4)), $(round(hi; sigdigits=4))]")
        g == r.groups[end] && name == r.model.param_names[end] || println(io)
    end
end

"""
    contrast(r::HierarchicalBayesianFitResults, group1, group2; param::Symbol) -> Vector{Float64}

Posterior samples of the native-scale difference
`exp(μ_pop[group1, param]) - exp(μ_pop[group2, param])`. Use `mean(out)`,
`quantile(out, [0.025, 0.975])`, and `mean(out .> 0)` for the standard
"is group1 higher than group2?" probability statement.
"""
function contrast(r::HierarchicalBayesianFitResults, g1::AbstractString, g2::AbstractString;
                  param::Symbol)
    g1 in r.groups || throw(ArgumentError("group `$g1` not present; have $(r.groups)"))
    g2 in r.groups || throw(ArgumentError("group `$g2` not present; have $(r.groups)"))
    String(param) in r.model.param_names ||
        throw(ArgumentError("`$param` not in model.param_names = $(r.model.param_names)"))
    s1 = Symbol("μ_pop_", g1, "_", param)
    s2 = Symbol("μ_pop_", g2, "_", param)
    return exp.(vec(r.chains[s1].data)) .- exp.(vec(r.chains[s2].data))
end

# Build the hierarchical Turing model. Non-centered reparameterization for NUTS
# efficiency. Population means are on the log scale; final per-curve params are
# `exp(μ_pop[group, k] + τ[k] * z[curve, k])`.
function build_hierarchical_turing_model(model::AbstractGrowthModel,
                                         priors_vec, sigma_prior, prior_tau,
                                         group_idx::Vector{Int}, likelihood::Symbol)
    n_params = length(priors_vec)
    n_groups = maximum(group_idx)
    n_curves = length(group_idx)

    # Convert LogNormal user priors to (μ_log, σ_log) tuples for the population layer.
    pop_means = [_lognormal_log_mean(p) for p in priors_vec]
    pop_sds   = [_lognormal_log_sd(p)   for p in priors_vec]

    if model isa NLModel
        return _build_hier_nl(model.func, n_params, n_groups, n_curves, group_idx,
                              pop_means, pop_sds, prior_tau, sigma_prior, likelihood)
    elseif model isa ODEModel
        return _build_hier_ode(model.func, model.n_eq, n_params, n_groups, n_curves, group_idx,
                               pop_means, pop_sds, prior_tau, sigma_prior, likelihood)
    else
        throw(ArgumentError("hierarchical fitting not supported for $(typeof(model))"))
    end
end

_lognormal_log_mean(p::LogNormal) = p.μ
_lognormal_log_sd(p::LogNormal)   = p.σ
_lognormal_log_mean(p) = throw(ArgumentError(
    "hierarchical fitting requires LogNormal priors on each parameter; got $(typeof(p))"))
_lognormal_log_sd(p)   = _lognormal_log_mean(p)

function _build_hier_nl(curve_func, n_params, n_groups, n_curves, group_idx,
                        pop_means, pop_sds, prior_tau, sigma_prior, likelihood)
    if likelihood === :lognormal
        return @model function _hier_nl_lognormal(times, ys)
            μ_pop ~ arraydist([Normal(pop_means[k], pop_sds[k]) for g in 1:n_groups, k in 1:n_params])
            τ     ~ arraydist([truncated(Normal(0, prior_tau); lower=0) for _ in 1:n_params])
            z     ~ arraydist([Normal(0, 1) for _ in 1:n_curves, _ in 1:n_params])
            σ_obs ~ filldist(sigma_prior, n_curves)
            for i in 1:n_curves
                g = group_idx[i]
                p_i = [exp(μ_pop[g, k] + τ[k] * z[i, k]) for k in 1:n_params]
                pred = curve_func(p_i, times)
                for j in eachindex(ys[i])
                    ys[i][j] ~ LogNormal(log(pred[j] + eps()), σ_obs[i])
                end
            end
        end
    elseif likelihood === :normal
        return @model function _hier_nl_normal(times, ys)
            μ_pop ~ arraydist([Normal(pop_means[k], pop_sds[k]) for g in 1:n_groups, k in 1:n_params])
            τ     ~ arraydist([truncated(Normal(0, prior_tau); lower=0) for _ in 1:n_params])
            z     ~ arraydist([Normal(0, 1) for _ in 1:n_curves, _ in 1:n_params])
            σ_obs ~ filldist(sigma_prior, n_curves)
            for i in 1:n_curves
                g = group_idx[i]
                p_i = [exp(μ_pop[g, k] + τ[k] * z[i, k]) for k in 1:n_params]
                pred = curve_func(p_i, times)
                for j in eachindex(ys[i])
                    ys[i][j] ~ Normal(pred[j], σ_obs[i])
                end
            end
        end
    else  # :proportional
        return @model function _hier_nl_proportional(times, ys)
            μ_pop ~ arraydist([Normal(pop_means[k], pop_sds[k]) for g in 1:n_groups, k in 1:n_params])
            τ     ~ arraydist([truncated(Normal(0, prior_tau); lower=0) for _ in 1:n_params])
            z     ~ arraydist([Normal(0, 1) for _ in 1:n_curves, _ in 1:n_params])
            σ_obs ~ filldist(sigma_prior, n_curves)
            for i in 1:n_curves
                g = group_idx[i]
                p_i = [exp(μ_pop[g, k] + τ[k] * z[i, k]) for k in 1:n_params]
                pred = curve_func(p_i, times)
                for j in eachindex(ys[i])
                    ys[i][j] ~ Normal(pred[j], σ_obs[i] * (pred[j] + eps()))
                end
            end
        end
    end
end

function _build_hier_ode(ode_func!, n_eq, n_params, n_groups, n_curves, group_idx,
                         pop_means, pop_sds, prior_tau, sigma_prior, likelihood)
    if likelihood === :lognormal
        return @model function _hier_ode_lognormal(times, ys)
            μ_pop ~ arraydist([Normal(pop_means[k], pop_sds[k]) for g in 1:n_groups, k in 1:n_params])
            τ     ~ arraydist([truncated(Normal(0, prior_tau); lower=0) for _ in 1:n_params])
            z     ~ arraydist([Normal(0, 1) for _ in 1:n_curves, _ in 1:n_params])
            σ_obs ~ filldist(sigma_prior, n_curves)
            for i in 1:n_curves
                g = group_idx[i]
                p_i = [exp(μ_pop[g, k] + τ[k] * z[i, k]) for k in 1:n_params]
                pred = _solve_ode_sum(ode_func!, p_i, times, ys[i][1], n_eq)
                if pred === nothing
                    Turing.@addlogprob! -Inf
                    return nothing
                end
                for j in eachindex(ys[i])
                    ys[i][j] ~ LogNormal(log(pred[j] + eps()), σ_obs[i])
                end
            end
        end
    elseif likelihood === :normal
        return @model function _hier_ode_normal(times, ys)
            μ_pop ~ arraydist([Normal(pop_means[k], pop_sds[k]) for g in 1:n_groups, k in 1:n_params])
            τ     ~ arraydist([truncated(Normal(0, prior_tau); lower=0) for _ in 1:n_params])
            z     ~ arraydist([Normal(0, 1) for _ in 1:n_curves, _ in 1:n_params])
            σ_obs ~ filldist(sigma_prior, n_curves)
            for i in 1:n_curves
                g = group_idx[i]
                p_i = [exp(μ_pop[g, k] + τ[k] * z[i, k]) for k in 1:n_params]
                pred = _solve_ode_sum(ode_func!, p_i, times, ys[i][1], n_eq)
                if pred === nothing
                    Turing.@addlogprob! -Inf
                    return nothing
                end
                for j in eachindex(ys[i])
                    ys[i][j] ~ Normal(pred[j], σ_obs[i])
                end
            end
        end
    else  # :proportional
        return @model function _hier_ode_proportional(times, ys)
            μ_pop ~ arraydist([Normal(pop_means[k], pop_sds[k]) for g in 1:n_groups, k in 1:n_params])
            τ     ~ arraydist([truncated(Normal(0, prior_tau); lower=0) for _ in 1:n_params])
            z     ~ arraydist([Normal(0, 1) for _ in 1:n_curves, _ in 1:n_params])
            σ_obs ~ filldist(sigma_prior, n_curves)
            for i in 1:n_curves
                g = group_idx[i]
                p_i = [exp(μ_pop[g, k] + τ[k] * z[i, k]) for k in 1:n_params]
                pred = _solve_ode_sum(ode_func!, p_i, times, ys[i][1], n_eq)
                if pred === nothing
                    Turing.@addlogprob! -Inf
                    return nothing
                end
                for j in eachindex(ys[i])
                    ys[i][j] ~ Normal(pred[j], σ_obs[i] * (pred[j] + eps()))
                end
            end
        end
    end
end

# Rename Turing's index-based names to human-readable group/param/curve labels.
function _rename_hier_params(chains, model::AbstractGrowthModel,
                             group_labels::Vector{String}, groups::Vector{String})
    n_params  = length(model.param_names)
    n_groups  = length(groups)
    n_curves  = length(group_labels)
    renames = Pair{String,String}[]
    for g in 1:n_groups, k in 1:n_params
        push!(renames, "μ_pop[$g, $k]" => "μ_pop_$(groups[g])_$(model.param_names[k])")
    end
    for k in 1:n_params
        push!(renames, "τ[$k]" => "τ_$(model.param_names[k])")
    end
    for i in 1:n_curves, k in 1:n_params
        push!(renames, "z[$i, $k]" => "z_$(group_labels[i])_$(i)_$(model.param_names[k])")
    end
    for i in 1:n_curves
        push!(renames, "σ_obs[$i]" => "σ_$(group_labels[i])_$(i)")
    end
    return replacenames(chains, renames...)
end
