# KM3OpenOsc Example: Plotting KM3NeT Open Test Data

This project demonstrates how to use `KM3OpenOsc.jl`  to load and analyze the KM3NeT ORCA 433 kt-y open data release using its test version. The example is provided as a [Pluto.jl](https://github.com/fonsp/Pluto.jl) notebook for interactivity and reproducibility.

## Example Notebook

The rendered version of the notebook can be found in the documentation [**here**](https://km3openosc-jl-fd9d00.pages.km3net.de/dev/notebooks/example_read_and_plot/).

The notebook walks through the following:
- Loading open data and binning definitions
- Using `KM3io.jl` to read `.root` files containing the data
- Computing oscillation matrices for oscillation parameters from the ORCA analysis paper
- Loading the Honda atmospheric neutrino flux from `NuFlux.jl`
- Filling 2D histograms with event categories using `KM3OpenOsc.jl`
- Exporting results to HDF5
- Plotting reconstructed and true event distributions

## Dependencies

This notebook uses the following Julia packages:
- `KM3io.jl`: KM3NeT package for I/O routines related to KM3NeT ROOT files.
- `KM3OpenOsc.jl`: Package for histogram filling for oscillation analysis.
- `NuFlux.jl`: Package to compute the atmospheric neutrino flux.
- `KM3NeTTestData.jl`: Package to load KM3NeT test data.
- `FHist.jl`: Package to perform histogram operations.
- `CairoMakie.jl`: Package for ploting purposes.

To run the notebook, install these packages in your Julia environment.

## Running the Notebook

1. Clone this repository.
2. Open Julia and install Pluto if needed:

```julia
import Pkg
Pkg.add("Pluto")
using Pluto
Pluto.run()
```

Then open the `example_read_and_plot.jl` file in your browser.