"""
    check_likelihood_data!(likelihood::Symbol, y::AbstractVector)

Guard for likelihood/data compatibility. `:lognormal` requires strictly positive `y`.
No silent epsilon shift — fail loud so the user fixes preprocessing.
"""
function check_likelihood_data!(likelihood::Symbol, y::AbstractVector)
    if likelihood === :lognormal && any(<=(0), y)
        throw(ArgumentError(
            "likelihood=:lognormal requires strictly positive data; got $(count(<=(0), y)) " *
            "non-positive value(s). Run `preprocess` with `correct_negatives=true`, or pass " *
            "`BayesFitOptions(likelihood=:normal)`."
        ))
    elseif likelihood !== :lognormal && likelihood !== :normal
        throw(ArgumentError("unknown likelihood $(likelihood); supported: :lognormal, :normal"))
    end
    return nothing
end
