using Documenter, KM3OpenOsc
#import Pkg; Pkg.add("PlutoStaticHTML")
#using PlutoStaticHTML
#
#const NOTEBOOK_DIR = joinpath(@__DIR__, "src","notebooks")
#
#function build()
#    println("Building notebooks in $NOTEBOOK_DIR")
#    oopts = OutputOptions(; append_build_context=true)
#    output_format = documenter_output
#    bopts = BuildOptions(NOTEBOOK_DIR; output_format)
#    build_notebooks(bopts, oopts)
#    return nothing
#end
#
#build()

makedocs(;
    modules = [KM3OpenOsc],
    sitename = "KM3OpenOsc.jl",
    authors = "Santiago Pena Martinez",
    format = Documenter.HTML(;
        assets = ["assets/custom.css"],
        sidebar_sitename = true,
        collapselevel = 2,
        warn_outdated = true,
       size_threshold = 1000_000,
    ),
    warnonly = [:missing_docs],
    pages = [
        "Home" => "index.md",
        "API" => "api.md",
        "Example with open data" => "notebooks/example_read_and_plot_open_data.md",
        #"Example with test data" => "notebooks/example_read_and_plot.md"
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
