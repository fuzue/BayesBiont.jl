# [Install](@id install)

```@contents
Pages = ["index.md"]
Depth = 2
```

## Requirements

- Julia 1.11 or newer.
- Kinbiont.jl 1.3+ (BayesBiont reuses Kinbiont's `GrowthData`, `ModelSpec`, and `MODEL_REGISTRY`).
- Turing.jl + Distributions.jl + MCMCChains.jl (pulled in automatically).

## Install from the General registry

```julia
using Pkg
Pkg.add("BayesBiont")
```

This will install BayesBiont, Kinbiont, Turing, and the rest of the dependency tree.

## Optional: SciMLSensitivity for ReverseDiff on ODE models

BayesBiont supports `adbackend=:reversediff` for hierarchical fits. For ODE models specifically, this requires `SciMLSensitivity.jl` to be loaded in your session because reverse-mode AD through SciML ODE solvers needs adjoint methods. It is **not** a hard dependency of BayesBiont (its Enzyme constraints can conflict with other packages in some configurations).

To opt in:

```julia
using Pkg
Pkg.add("SciMLSensitivity")
using SciMLSensitivity
using BayesBiont
# Now `BayesFitOptions(adbackend=:reversediff)` works on ODE models too.
```

If you try ODE + `:reversediff` without `SciMLSensitivity` loaded, BayesBiont raises an `ArgumentError` with a clear hint.

For NL closed-form models (logistic, Gompertz, Richards, etc.) `:reversediff` works out of the box, no SciMLSensitivity needed.

## Development install

To work against a local development tree (e.g. to modify BayesBiont source or to track an unreleased Kinbiont):

```julia
using Pkg
Pkg.develop(path="path/to/BayesBiont.jl")
# Optional: also develop Kinbiont
Pkg.develop(path="path/to/Kinbiont.jl")
```

## Verify the install

```julia
using BayesBiont, Kinbiont
@assert isdefined(BayesBiont, :bayesfit)
@assert haskey(Kinbiont.MODEL_REGISTRY, "NL_Gompertz")
```
