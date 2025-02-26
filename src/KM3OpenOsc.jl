module KM3OpenOsc

import Base: read, write
using JSON
import UnROOT
using KM3io
using FHist
using Neurthino
using NuFlux

include("exports.jl")

@template (FUNCTIONS, METHODS, MACROS) =
    """
    $(TYPEDSIGNATURES)
    $(DOCSTRING)
    """

@template TYPES = """
    $(TYPEDEF)

    $(DOCSTRING)

    # Fields
    $(TYPEDFIELDS)
    """

include("root/osc_opendata.jl")

end # module