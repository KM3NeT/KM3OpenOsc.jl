# KM3OpenOsc.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://common.pages.km3net.de/KM3OpenOsc.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://common.pages.km3net.de/KM3OpenOsc.jl/dev)
[![Build Status](https://git.km3net.de/common/KM3OpenOsc.jl/badges/main/pipeline.svg)](https://git.km3net.de/common/KM3OpenOsc.jl/pipelines)
[![Coverage](https://git.km3net.de/common/KM3OpenOsc.jl/badges/main/coverage.svg)](https://git.km3net.de/common/KM3OpenOsc.jl/commits/main)

Welcome to the `KM3OpenOsc.jl` repository!


## Documentation

Check out the **[Latest Documention](https://common.pages.km3net.de/KM3OpenOsc.jl/dev)**
which also includes tutorials and examples.


## Installation

`KM3OpenOsc.jl` is **not an officially registered Julia package** but it's available via the **[KM3NeT Julia registry](https://git.km3net.de/common/julia-registry)**. To add the KM3NeT Julia registry to your local Julia registry list, follow the instructions in its
[README](https://git.km3net.de/common/julia-registry#adding-the-registry) or simply do

    git clone https://git.km3net.de/common/julia-registry ~/.julia/registries/KM3NeT
    
After that, you can add `KM3OpenOsc.jl` just like any other Julia package:

    julia> import Pkg; Pkg.add("KM3OpenOsc")
    

## Quickstart

``` julia-repl
julia> using KM3OpenOsc
```
