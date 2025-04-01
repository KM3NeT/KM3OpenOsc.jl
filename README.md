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

## Quick example with KM3NeT Test data

```julia
using KM3OpenOsc
using KM3io
using KM3NeTTestData
using NuFlux

OSCFILE = KM3NeTTestData.datapath("oscillations", "ORCA6_433kt-y_opendata_v0.5_testdata.root")
BINDEF = KM3NeTTestData.datapath("oscillations", "bins_433kt-y_v0.4.json")

f = KM3io.OSCFile(OSCFILE)
nu = f.osc_opendata_nu
data = f.osc_opendata_data

hn = create_histograms(BINDEF)
hd = create_histograms(BINDEF)

BF = Dict("dm_21" => 7.42e-5, #ORCA 433 kt-y standard oscillations Best Fit
                       "dm_31" => 2.18e-3,
                       "theta_12" => deg2rad(33.45),
                       "theta_23" => deg2rad(45.57299599919429),
                       "theta_13" => deg2rad(8.62),
                       "dcp" => deg2rad(230))

NUFLUX_PATH = split(Base.pathof(NuFlux), "src")[1]
FLUX_DATA_DIR = joinpath(NUFLUX_PATH, "data")
flux_path = joinpath(FLUX_DATA_DIR, "frj-ally-20-12-solmin.d") # Get flux of Honda Frejus site from NuFlux

flux_dict = get_flux_dict(flux_path) # If no flux path is provided, Honda flux at frejus site is taken by default
U,H = get_oscillation_matrices(BF) # If no dict of parameters is provided NuFit is selected by default

fill_response!(hn, nu, flux_dict, U, H; oscillations=true, livetime=1.39) # fill neutrinos ,need flux, oscillation parameters and livetime
fill_response!(hd, data) # fill data, don't need to specify much
export_histograms_hdf5(hn, "neutrino_histograms_from_testdata.h5") # You can easily export the filled histograms to hdf5
h5f = build_HDF5_file("responses_to_file.h5") # Create h5 file with same structure as responses bins 
fill_HDF5_file!(h5f, nu, hn, "neutrinos") # Completely export the response as a table in an hdf5 file at a given path 
fill_HDF5_file!(h5f, data, hd, "data") # Completely export the response as a table in an hdf5 file at a given path 
close(h5f)
```
