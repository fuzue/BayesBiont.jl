using Kinbiont: GrowthData

"""
    group_from_labels(data::GrowthData; pattern=r"^([A-Za-z]+)") -> Vector{String}

Parse a grouping vector from `data.labels` by extracting the first regex capture
group from each label. Default pattern grabs the leading alphabetic prefix
(e.g. `"WT_1"` → `"WT"`). Helper for v0.2 hierarchical pooling; usable today for
ad-hoc grouping.
"""
function group_from_labels(data::GrowthData; pattern::Regex=r"^([A-Za-z]+)")
    return map(data.labels) do label
        m = match(pattern, label)
        m === nothing && throw(ArgumentError("label `$label` does not match pattern $pattern"))
        String(m.captures[1])
    end
end
