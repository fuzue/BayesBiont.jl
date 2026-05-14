using Documenter, BayesBiont

push!(LOAD_PATH, "../src/")

makedocs(
    sitename = "BayesBiont.jl Documentation",
    modules  = [BayesBiont],
    source   = "src/",
    checkdocs = :none,
    format = Documenter.HTML(
        size_threshold = nothing,
    ),
    pages = [
        "Home"             => "index.md",
        "Install"          => "01_install/index.md",
        "Quick start"      => "02_quickstart/index.md",
        "Models"           => "03_models/index.md",
        "Priors"           => "04_priors/index.md",
        "Hierarchical"     => "05_hierarchical/index.md",
        "Model comparison" => "06_comparison/index.md",
        "Diagnostics"      => "07_diagnostics/index.md",
        "API reference"    => "api.md",
    ],
)
