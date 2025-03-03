using Documenter, KM3OpenOsc

makedocs(;
    modules = [KM3OpenOsc],
    sitename = "KM3OpenOsc.jl",
    authors = "Santiago Pena Martinez",
    format = Documenter.HTML(;
        assets = ["assets/custom.css"],
        sidebar_sitename = true,
        collapselevel = 2,
        warn_outdated = true,
    ),
    warnonly = [:missing_docs],
    pages = [
        "Home" => "index.md",
        "API" => "api.md"
        #"Examples" => "examples/an_example.md"
    ],
    repo = Documenter.Remotes.URL(
        "https://git.km3net.de/common/KM3OpenOsc.jl/blob/{commit}{path}#L{line}",
        "https://git.km3net.de/common/KM3OpenOsc.jl"
    ),
)

deploydocs(;
  repo = "git.km3net.de/common/KM3OpenOsc.jl",
  devbranch = "main",
  push_preview=true
)
