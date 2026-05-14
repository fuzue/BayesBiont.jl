using Documenter, BayesBiont

makedocs(
    sitename = "BayesBiont.jl Documentation",
    modules  = [BayesBiont],
    checkdocs = :none,
    format = Documenter.HTML(
        size_threshold = nothing,
        canonical = "https://fuzue.github.io/BayesBiont.jl/",
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

deploydocs(
    repo = "github.com/fuzue/BayesBiont.jl.git",
    devbranch = "main",
    push_preview = false,
)
