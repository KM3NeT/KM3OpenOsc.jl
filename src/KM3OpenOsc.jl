module KM3OpenOsc

import Base: read, write
using JSON
import UnROOT
using KM3io
using FHist
using Neurthino
using NuFlux
using Corpuscles

include("exports.jl")
include("root/osc_opendata.jl")

end # module