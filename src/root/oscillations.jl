struct OSCFile
    _fobj::Union{UnROOT.ROOTFile,Dict}
    rawroot::Union{UnROOT.ROOTFile,Nothing}
    osc_opendata_nu::Union{OscOpenDataTree,Nothing}
    osc_opendata_data::Union{OscOpenDataTree,Nothing}
    osc_opendata_muons::Union{OscOpenDataTree,Nothing}

    function OSCFile(filename::AbstractString)
        if endswith(filename, ".root")
            fobj = UnROOT.ROOTFile(filename)
            osc_opendata_nu = ROOT.TTREE_OSC_OPENDATA_NU ∈ keys(fobj) ? OscOpenDataTree(fobj, ROOT.TTREE_OSC_OPENDATA_NU) : nothing
            osc_opendata_data = ROOT.TTREE_OSC_OPENDATA_DATA ∈ keys(fobj) ? OscOpenDataTree(fobj, ROOT.TTREE_OSC_OPENDATA_DATA) : nothing
            osc_opendata_muons = ROOT.TTREE_OSC_OPENDATA_MUONS ∈ keys(fobj) ? OscOpenDataTree(fobj, ROOT.TTREE_OSC_OPENDATA_MUONS) : nothing
            return new(fobj, fobj, osc_opendata_nu, osc_opendata_data, osc_opendata_muons)
        end
    end
end
Base.close(f::OSCFile) = close(f._fobj)
function Base.show(io::IO, f::OSCFile)
    if isa(f._fobj, UnROOT.ROOTFile)
        s = String[]
        !isnothing(f.osc_opendata_nu) && push!(s, "$(f.osc_opendata_nu)")
        !isnothing(f.osc_opendata_data) && push!(s, "$(f.osc_opendata_data)")
        !isnothing(f.osc_opendata_muons) && push!(s, "$(f.osc_opendata_muons)")
        info = join(s, ", ")
        print(io, "OSCFile{$info}")
    else
        print(io, "Empty OSCFile")
    end

end
