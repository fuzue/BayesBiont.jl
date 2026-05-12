using Random
using OrdinaryDiffEqTsit5: Tsit5
using SciMLBase: ODEProblem, solve
using Kinbiont: MODEL_REGISTRY

"""
    synthetic_gompertz(; N_max, growth_rate, lag, σ, t_end=24.0, dt=0.2, seed=42)

Generate a single synthetic Gompertz curve with multiplicative log-normal noise.
Returns `(times::Vector, observed::Vector, truth::NamedTuple)`.
"""
function synthetic_gompertz(;
    N_max = 1.0,
    growth_rate = 0.4,
    lag = 5.0,
    σ = 0.05,
    t_end = 24.0,
    dt = 0.2,
    seed = 42,
)
    rng = MersenneTwister(seed)
    times = collect(0.0:dt:t_end)
    clean = N_max .* exp.(-exp.(-growth_rate .* (times .- lag)))
    observed = clean .* exp.(σ .* randn(rng, length(times)))
    truth = (; N_max, growth_rate, lag, σ)
    return times, observed, truth
end

"""
    synthetic_logistic(; K, N_0, r, σ, t_end=24.0, dt=0.2, seed=23)

Generate a single synthetic logistic curve matching `Kinbiont.NL_model_logistic`'s
positional parameterization `p = [p1, p2, p3]` with `p1=K, p2=N_0, p3=r`. Note that
Kinbiont labels these as `["N_max", "growth_rate", "lag"]` in the registry — those
names are misleading for this model; positional semantics is what matters here.
"""
function synthetic_logistic(;
    K = 1.0,
    N_0 = 0.01,
    r = 0.5,
    σ = 0.05,
    t_end = 24.0,
    dt = 0.2,
    seed = 23,
)
    rng = MersenneTwister(seed)
    times = collect(0.0:dt:t_end)
    clean = K ./ (1 .+ (K / N_0 - 1) .* exp.(-r .* times))
    observed = clean .* exp.(σ .* randn(rng, length(times)))
    truth = (; K, N_0, r, σ)
    return times, observed, truth
end

"""
    synthetic_ahpm(; gr, exit_lag_rate, N_max, shape, N0, σ, t_end=24.0, dt=0.2, seed=31)

Generate an observed total-cell curve from Kinbiont's aHPM ODE (2-state). Initial
state `[N0, 0]` — all dormant — matches BayesBiont's u0 convention.
Returns `(times, observed, truth)` with multiplicative noise.
"""
function synthetic_ahpm(;
    gr = 0.5,
    exit_lag_rate = 0.3,
    N_max = 1.0,
    shape = 1.0,
    N0 = 0.05,
    σ = 0.05,
    t_end = 24.0,
    dt = 0.25,
    seed = 31,
)
    rng = MersenneTwister(seed)
    times = collect(0.0:dt:t_end)
    p = [gr, exit_lag_rate, N_max, shape]
    u0 = [N0, 0.0]
    prob = ODEProblem(MODEL_REGISTRY["aHPM"].func, u0, (times[1], times[end]), p)
    sol = solve(prob, Tsit5(); saveat=times, abstol=1e-9, reltol=1e-7)
    clean = vec(sum(reduce(hcat, sol.u); dims=1))
    observed = clean .* exp.(σ .* randn(rng, length(times)))
    truth = (; gr, exit_lag_rate, N_max, shape, N0, σ)
    return times, observed, truth
end
