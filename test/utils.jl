using Random

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
