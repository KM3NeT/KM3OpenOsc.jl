"""

A response matrix bin.

"""

const NUE_PDGID = Particle("nu(e)0").pdgid.value
const ANUE_PDGID = Particle("~nu(e)0").pdgid.value
const NUMU_PDGID = Particle("nu(mu)0").pdgid.value
const ANUMU_PDGID = Particle("~nu(mu)0").pdgid.value
const NUTAU_PDGID = Particle("nu(tau)0").pdgid.value
const ANUTAU_PDGID = Particle("~nu(tau)0").pdgid.value

struct ResponseMatrixBin
    E_reco_bin::Int64
    Ct_reco_bin::Int64

    E_true_bin::Union{Int64, Missing}
    Ct_true_bin::Union{Int64, Missing}

    Flav::Union{Int16, Missing}
    IsCC::Union{Int16, Missing}

    AnaClass::Int16

    W::Float64
    Werr::Union{Float64, Missing}
end

struct ResponseMatrixBinNeutrinos
    E_reco_bin::Float64
    Ct_reco_bin::Float64

    E_true_bin::Float64
    Ct_true_bin::Float64

    Flav::Int16
    IsCC::Int16

    AnaClass::Int16

    W::Float64
    Werr::Float64
end

struct ResponseMatrixBinMuons
    E_reco_bin::Float64
    Ct_reco_bin::Float64

    AnaClass::Int16

    W::Float64
    Werr::Float64
end

struct ResponseMatrixBinData
    E_reco_bin::Float64
    Ct_reco_bin::Float64

    AnaClass::Int16

    W::Float64
end



function _getpdgnumber(flav::Integer, isNB::Integer)
    flav == 0 && isNB == 0 && return NUE_PDGID # nu(e)0
    flav == 0 && isNB == 1 && return ANUE_PDGID # ~nu(e)0
    flav == 1 && isNB == 0 && return NUMU_PDGID # nu(mu)0
    flav == 1 && isNB == 1 && return ANUMU_PDGID # ~nu(mu)0
    flav == 2 && isNB == 0 && return NUTAU_PDGID # nu(tau)0
    flav == 2 && isNB == 1 && return ANUTAU_PDGID # ~nu(tau)0

    error("Invalid flavor: $flav($isNB)")
end

function _getanaclassname(fClass::Integer)
    fClass == 1 && return "HighPurityTracks"
    fClass == 2 && return "Showers"
    fClass == 3 && return "LowPurityTracks"
    error("Invalid class: $fClass)")
end



"""
OscOpenDataTree definitions for structure.
"""
struct OscOpenDataTree{T} <: KM3io.OscillationsData
    _fobj::UnROOT.ROOTFile
    #header::Union{MCHeader, Missing} # no header for now, subject to change
    _bin_lookup_map::Dict{Tuple{Int,Int,Int},Int} # Not implemented for now
    _t::T  # carry the type to ensure type-safety
    tpath::String

    function OscOpenDataTree(fobj::UnROOT.ROOTFile, tpath::String)
        if tpath == ROOT.TTREE_OSC_OPENDATA_NU
            branch_paths = ["E_reco_bin",
                "Ct_reco_bin",
                "Flav",
                "IsCC",
                "IsNB",
                "E_true_bin",
                "Ct_true_bin",
                "W",
                "WE",
                "Class",
            ]

        elseif tpath == ROOT.TTREE_OSC_OPENDATA_DATA
            branch_paths = ["E_reco_bin",
            "Ct_reco_bin",
            "W",
            "Class",
            ]
        elseif tpath == ROOT.TTREE_OSC_OPENDATA_MUONS 
            branch_paths = ["E_reco_bin",
            "Ct_reco_bin",
            "W",
            "Class",
            ]
        end


        t = UnROOT.LazyTree(fobj, tpath, branch_paths)

        new{typeof(t)}(fobj, Dict{Tuple{Int,Int,Int},Int}(), t, tpath)
    end
end

OscOpenDataTree(filename::AbstractString, tpath::String) = OscOpenDataTree(UnROOT.ROOTFile(filename), tpath)

Base.close(f::OscOpenDataTree) = close(f._fobj)
Base.length(f::OscOpenDataTree) = length(f._t)
Base.firstindex(f::OscOpenDataTree) = 1
Base.lastindex(f::OscOpenDataTree) = length(f)
Base.eltype(::OscOpenDataTree) = ResponseMatrixBin
function Base.iterate(f::OscOpenDataTree, state=1)
    state > length(f) ? nothing : (f[state], state + 1)
end
function Base.show(io::IO, f::OscOpenDataTree)
    print(io, "OscOpenDataTree ($(length(f)) events)")
end

Base.getindex(f::OscOpenDataTree, r::UnitRange) = [f[idx] for idx ∈ r]
Base.getindex(f::OscOpenDataTree, mask::BitArray) = [f[idx] for (idx, selected) ∈ enumerate(mask) if selected]
function Base.getindex(f::OscOpenDataTree, idx::Integer)
    if idx > length(f)
        throw(BoundsError(f, idx))
    end
    idx > length(f) && throw(BoundsError(f, idx))
    e = f._t[idx]  # the event as NamedTuple: struct of arrays

    ResponseMatrixBin(
        e.E_reco_bin,
        e.Ct_reco_bin,
        get(e, :E_true_bin, missing),
        get(e, :Ct_true_bin, missing),
        get(e, :Flav, -99) > -99 ? _getpdgnumber(e.Flav, e.IsNB) : missing,
        get(e, :IsCC, missing), 
        e.Class, 
        e.W,
        get(e, :WE, missing),)
end

"""
Histogram definitions structure.
"""
struct HistogramDefinitions
    xbins::Union{Int64,Vector{Float64}}
    ybins::Union{Int64,Vector{Float64},Nothing}
end

struct HistogramsOscillations
    hists_true::Dict{String,Hist2D}
    hists_reco::Dict{String,Hist2D}

    function HistogramsOscillations(bins_true::HistogramDefinitions, bins_reco::HistogramDefinitions)
        hists_true = Dict{String,Hist2D}()
        hists_reco = Dict{String,Hist2D}()
        hist_reco_names = [
            "reco",
            "recoHighPurityTracks",
            "recoLowPurityTracks",
            "recoShowers",
        ]
        hist_true_names = [
            "elec_cc_nu",
            "elec_cc_nub",
            "nc_nu",
            "nc_nub",
            "muon_cc_nu",
            "muon_cc_nub",
            "tau_cc_nu",
            "tau_cc_nub",
            "true",
            "trueHighPurityTracks",
            "trueLowPurityTracks",
            "trueShowers",
        ]
        for n in hist_true_names
            hists_true[n] = _build_hist(bins_true.xbins, bins_true.ybins)
        end
        for n in hist_reco_names
            hists_reco[n] = _build_hist(bins_reco.xbins, bins_reco.ybins)
        end

        new(hists_true, hists_reco)
    end


    function _build_hist(xbins::Union{Int64,Vector{Float64}}, ybins::Union{Int64,Vector{Float64}})
        return Hist2D(; binedges=(xbins, ybins))
    end
end

function export_histograms_hdf5(histo::HistogramsOscillations, filename::String)
    for (name, hist) in histo.hists_true
        h5writehist(filename, "hists_true/"*name, hist)
    end
    for (name, hist) in histo.hists_reco
        h5writehist(filename, "hists_reco/"*name, hist)
    end
end

function fill_hist_by_bin!(h::Hist2D, xbin::Int64, ybin::Int64, w::Float64, werr::Float64) # This is a bit of a hacky way to fill the histograms, but it works
    bincounts(h)[xbin, ybin] += w
    sumw2(h)[xbin, ybin] += werr
end




function fill_all_hists_from_event!(hs::HistogramsOscillations, e::ResponseMatrixBin; livetime::Float64=1.)
    W = e.W * livetime
    Werr = e.Werr * livetime^2
    if ismissing(e.Werr)
        Werr = e.W^2
    end
    if ! ismissing(e.IsCC)
        if Bool(e.IsCC)
            if Particle(e.Flav).pdgid.value == Particle("nu(e)0").pdgid.value
                fill_hist_by_bin!(hs.hists_true["elec_cc_nu"], e.E_true_bin, e.Ct_true_bin, W, Werr)
            elseif Particle(e.Flav).pdgid.value == Particle("~nu(e)0").pdgid.value
                fill_hist_by_bin!(hs.hists_true["elec_cc_nub"], e.E_true_bin, e.Ct_true_bin, W, Werr)
            elseif Particle(e.Flav).pdgid.value == Particle("nu(mu)0").pdgid.value
                fill_hist_by_bin!(hs.hists_true["muon_cc_nu"], e.E_true_bin, e.Ct_true_bin, W, Werr)
            elseif Particle(e.Flav).pdgid.value == Particle("~nu(mu)0").pdgid.value
                fill_hist_by_bin!(hs.hists_true["muon_cc_nub"], e.E_true_bin, e.Ct_true_bin, W, Werr)
            elseif Particle(e.Flav).pdgid.value == Particle("nu(tau)0").pdgid.value
                fill_hist_by_bin!(hs.hists_true["tau_cc_nu"], e.E_true_bin, e.Ct_true_bin, W, Werr)
            elseif Particle(e.Flav).pdgid.value == Particle("~nu(tau)0").pdgid.value
                fill_hist_by_bin!(hs.hists_true["tau_cc_nub"], e.E_true_bin, e.Ct_true_bin, W, Werr)
            end
        else
            if Particle(e.Flav).pdgid.value == Particle("nu(mu)0").pdgid.value
                fill_hist_by_bin!(hs.hists_true["nc_nu"], e.E_true_bin, e.Ct_true_bin, W, Werr)
            elseif Particle(e.Flav).pdgid.value == Particle("~nu(mu)0").pdgid.value
                fill_hist_by_bin!(hs.hists_true["nc_nub"], e.E_true_bin, e.Ct_true_bin, W, Werr)
            end
        end
    end
    if _getanaclassname(e.AnaClass) == "HighPurityTracks"
        if ! ismissing(e.IsCC)
            fill_hist_by_bin!(hs.hists_true["trueHighPurityTracks"], e.E_true_bin, e.Ct_true_bin, W, Werr)
        end
        fill_hist_by_bin!(hs.hists_reco["recoHighPurityTracks"], e.E_reco_bin, e.Ct_reco_bin, W, Werr)
    elseif _getanaclassname(e.AnaClass) == "LowPurityTracks"
        if ! ismissing(e.IsCC)
            fill_hist_by_bin!(hs.hists_true["trueLowPurityTracks"], e.E_true_bin, e.Ct_true_bin, W, Werr)
        end
        fill_hist_by_bin!(hs.hists_reco["recoLowPurityTracks"], e.E_reco_bin, e.Ct_reco_bin, W, Werr)
    elseif _getanaclassname(e.AnaClass) == "Showers"
        if ! ismissing(e.IsCC)
            fill_hist_by_bin!(hs.hists_true["trueShowers"], e.E_true_bin, e.Ct_true_bin, W, Werr)
        end
        fill_hist_by_bin!(hs.hists_reco["recoShowers"], e.E_reco_bin, e.Ct_reco_bin, W, Werr)
    end
    if ! ismissing(e.IsCC)
        fill_hist_by_bin!(hs.hists_true["true"], e.E_true_bin, e.Ct_true_bin, W, Werr)
    end
    fill_hist_by_bin!(hs.hists_reco["reco"], e.E_reco_bin, e.Ct_reco_bin, W, Werr)

end

function fill_response!(hs::HistogramsOscillations, f::OscOpenDataTree,  flux_dict::Union{Dict, Nothing}=nothing, U0::Union{Matrix{ComplexF64}, Nothing}=nothing, H0::Union{Vector{ComplexF64}, Nothing}=nothing; oscillations::Union{Bool, Nothing}=true, livetime::Union{Float64, Nothing}=1.)
    if f.tpath == ROOT.TTREE_OSC_OPENDATA_NU
        for e in f
           fill_all_hists_from_event_oscillations_and_flux!(hs, e, flux_dict, U0, H0; oscillations=oscillations, livetime=livetime)
        end
    else
        for e in f
            fill_all_hists_from_event!(hs, e; livetime=livetime)
        end
    end
end


#"""
#Flux parameters structure.
#"""
#function get_flux_dict()
#    NUFLUX_PATH = split(Base.pathof(NuFlux), "src")[1]
#    FLUX_DATA_DIR = joinpath(NUFLUX_PATH, "data")
#    honda_flux = NuFlux.readfluxfile(joinpath(FLUX_DATA_DIR, "frj-ally-20-12-solmin.d"))
#    return Dict(NUE_PDGID => honda_flux[3],
#        NUMU_PDGID => honda_flux[1],
#        ANUE_PDGID => honda_flux[4],
#        ANUMU_PDGID => honda_flux[2],)
#end

"""
Oscillation parameters structure.
"""
function get_oscillation_matrices(nu_params::Dict=Dict(
		"dm_21" => 7.42e-5,
		"dm_31" => 2.510e-3,
		"theta_12" => deg2rad(33.45),
		"theta_23" => deg2rad(42.1),
		"theta_13" => deg2rad(8.62),
		"dcp" => deg2rad(230)
	)) #Nufit default parameters
    osc = OscillationParameters(3)
	setΔm²!(osc, 1=>2, -1 * nu_params["dm_21"])
	setΔm²!(osc, 1=>3, -1 * nu_params["dm_31"])
	setθ!(osc, 1=>2, nu_params["theta_12"])
	setθ!(osc, 2=>3, nu_params["theta_23"])
	setθ!(osc, 1=>3, nu_params["theta_13"])
	setδ!(osc, 1=>3, nu_params["dcp"])

	U = PMNSMatrix(osc)
	H = Hamiltonian(osc)
    return (U, H)
end

function osc_weight_computation(E::Float64, zdir::Float64, Flav::Int16, IsCC::Int16, flux_dict::Dict,U0::Union{Matrix{ComplexF64}, Nothing}, H0::Union{Vector{ComplexF64},Nothing}, oscillations::Bool)
	weight = 0
	isAnti = (Flav<0)
	path = Neurthino.prempath(acos(zdir), 2, samples=20);
	osc_values = oscprob(U0, H0, E, path, anti = isAnti);
	
    #if oscillations
    #	for flav in [NUE_PDGID, NUMU_PDGID]
    #		nuin = flav < NUE_PDGID+1 ? 1 : 2
    #		if abs(Flav)==NUE_PDGID
    #			nuout=1
    #		elseif abs(Flav)==NUMU_PDGID
    #			nuout=2
    #		elseif abs(Flav)==NUTAU_PDGID
    #			nuout=3
    #		end
    #		flux_value = NuFlux.flux(flux_dict[flav * sign(Flav)], E, zdir)
    #		if (IsCC > 0)
    #			osc_prob_cc = osc_values[1,1,nuin, nuout]
    #			weight += flux_value*osc_prob_cc
    #		else
    #			weight += flux_value
    #		end
    #    end
    #else
    #    if abs(Flav)==NUTAU_PDGID
    #        weight = 0
    #    else
    #		if (IsCC > 0)
    #            flux_value = NuFlux.flux(flux_dict[Flav], E, zdir)
    #            weight += flux_value
    #        else
    #            for flav in [NUE_PDGID, NUMU_PDGID]
    #                flux_value = NuFlux.flux(flux_dict[flav * sign(Flav)], E, zdir)
    #                weight += flux_value
    #            end
    #        end
    #    end
    #end
	return weight
end

function fill_all_hists_from_event_oscillations_and_flux!(hs::HistogramsOscillations, e::ResponseMatrixBin, flux_dict::Dict, U0::Union{Matrix{ComplexF64}, Nothing}=nothing, H0::Union{Vector{ComplexF64}, Nothing}=nothing; oscillations::Bool=true, livetime::Float64=1.)

    if oscillations && (U0 == nothing || H0 == nothing)
        error("Oscillations are enabled, but no PMNS matrix or Hamiltonian was provided.")
    end
	
    E = (binedges(hs.hists_true["true"])[1][e.E_true_bin] .* binedges(hs.hists_true["true"])[1][e.E_true_bin+1]).^.5
	zdir = bincenters(hs.hists_true["true"])[2][e.Ct_true_bin]
	weight = osc_weight_computation(E, zdir, e.Flav, e.IsCC, flux_dict, U0, H0, oscillations)
	

	new_W = e.W * weight * livetime
	new_Werr = e.Werr * weight^2 * livetime^2
    new_e = ResponseMatrixBin(e.E_reco_bin, e.Ct_reco_bin,  e.E_true_bin, e.Ct_true_bin,  e.Flav, e.IsCC, e.AnaClass, new_W, new_Werr)

    fill_all_hists_from_event!(hs, new_e)

end

function create_histograms(fhist::UnROOT.ROOTFile)
    xbins_true = fhist["hbinstrue"][:fXaxis_fXbins]
	ybins_true =  fhist["hbinstrue"][:fYaxis_fXbins]
	xbins_reco = fhist["hbinsreco"][:fXaxis_fXbins]
	ybins_reco =  fhist["hbinsreco"][:fYaxis_fXbins]
    bins_true = HistogramDefinitions(xbins_true, ybins_true)
	bins_reco = HistogramDefinitions(xbins_reco, ybins_reco)
    return HistogramsOscillations(bins_true, bins_reco)
end


function create_histograms_from_json(fjson::String)
    hist_edges = JSON.parsefile(fjson)
    xbins_true = Float64.(hist_edges["E_true_bins"])
    ybins_true =  Float64.(hist_edges["cosT_true_bins"])
    xbins_reco = Float64.(hist_edges["E_reco_bins"])
    ybins_reco =  Float64.(hist_edges["cosT_reco_bins"])
    bins_true = HistogramDefinitions(xbins_true, ybins_true)
	bins_reco = HistogramDefinitions(xbins_reco, ybins_reco)
    return HistogramsOscillations(bins_true, bins_reco)
end


function fill_HDF5_file!(h5file::H5File, hs::HistogramsOscillations, e::ResponseMatrixBin, filetype::String="neutrinos")
    
    Ereco = bincenters(hs.hists_reco["reco"])[1][e.E_reco_bin]
    zdirreco = bincenters(hs.hists_reco["reco"])[2][e.Ct_reco_bin]
    if filetype=="neutrinos"
        Etrue = (binedges(hs.hists_true["true"])[1][e.E_true_bin] .* binedges(hs.hists_true["true"])[1][e.E_true_bin+1]).^.5
        zdirtrue = bincenters(hs.hists_true["true"])[2][e.Ct_true_bin]

        new_e = ResponseMatrixBinNeutrinos(Ereco, zdirreco, Etrue, zdirtrue, e.Flav, e.IsCC, e.AnaClass, e.W, e.Werr)
        if Bool(e.IsCC)
            if Particle(e.Flav).pdgid.value == Particle("nu(e)0").pdgid.value
                push!(h5file._datasets["elec_cc_nu"], new_e)
            elseif Particle(e.Flav).pdgid.value == Particle("~nu(e)0").pdgid.value
                push!(h5file._datasets["elec_cc_nub"], new_e)
            elseif Particle(e.Flav).pdgid.value == Particle("nu(mu)0").pdgid.value
                push!(h5file._datasets["muon_cc_nu"], new_e)
            elseif Particle(e.Flav).pdgid.value == Particle("~nu(mu)0").pdgid.value
                push!(h5file._datasets["muon_cc_nub"], new_e)
            elseif Particle(e.Flav).pdgid.value == Particle("nu(tau)0").pdgid.value
                push!(h5file._datasets["tau_cc_nu"], new_e)
            elseif Particle(e.Flav).pdgid.value == Particle("~nu(tau)0").pdgid.value
                push!(h5file._datasets["tau_cc_nub"], new_e)
            end
        else
            if Particle(e.Flav).pdgid.value == Particle("nu(mu)0").pdgid.value
                push!(h5file._datasets["nc_nu"], new_e)
            elseif Particle(e.Flav).pdgid.value == Particle("~nu(mu)0").pdgid.value
                push!(h5file._datasets["nc_nub"], new_e)
            end
        end
            	
    elseif filetype=="atm_muons"
        new_e = ResponseMatrixBinMuons(Ereco, zdirreco, e.AnaClass, e.W, e.Werr)
        push!(h5file._datasets["atm_muons"], new_e)
    elseif filetype=="data"
        new_e = ResponseMatrixBinData(Ereco, zdirreco, e.AnaClass, e.W)
        push!(h5file._datasets["data"], new_e)

    end
   
end

function build_HDF5_file(filename::String="PISA.h5")
    fh5 = H5File(filename, "w")
    true_pid = ["elec_cc_nu",
            "elec_cc_nub",
            "nc_nu",
            "nc_nub",
            "muon_cc_nu",
            "muon_cc_nub",
            "tau_cc_nu",
            "tau_cc_nub",
    ]
    for pid in true_pid
        #create_group(fh5, pid)
        KM3io.create_dataset(fh5, pid, ResponseMatrixBinNeutrinos)
    end
    KM3io.create_dataset(fh5, "atm_muons", ResponseMatrixBinMuons)
    println(KM3io.create_dataset(fh5, "data", ResponseMatrixBinData))
    return fh5
end
