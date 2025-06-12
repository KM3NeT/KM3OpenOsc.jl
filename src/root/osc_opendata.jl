const NUE_PDGID = Particle("nu(e)0").pdgid.value
const ANUE_PDGID = Particle("~nu(e)0").pdgid.value
const NUMU_PDGID = Particle("nu(mu)0").pdgid.value
const ANUMU_PDGID = Particle("~nu(mu)0").pdgid.value
const NUTAU_PDGID = Particle("nu(tau)0").pdgid.value
const ANUTAU_PDGID = Particle("~nu(tau)0").pdgid.value


"""

`HistogramDefinitions` is a structure defining the binning for histograms.

"""
struct HistogramDefinitions
    xbins::Union{Int64,Vector{Float64}}
    ybins::Union{Int64,Vector{Float64},Nothing}
end

"""

`HistogramsOscillations`is a structure containing histograms for oscillation analysis.

"""
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
end

function _build_hist(xbins::Union{Int64,Vector{Float64}}, ybins::Union{Int64,Vector{Float64}})
    return Hist2D(; binedges=(xbins, ybins))
end


"""

Export histograms to an HDF5 file.

"""
function export_histograms_hdf5(histo::HistogramsOscillations, filename::String)
    if isfile(filename)
        @warn "File $filename already exists. It will be overwritten."
        rm(filename)
    end
    for (name, hist) in histo.hists_true
        h5writehist(filename, "hists_true/"*name, hist)
    end
    for (name, hist) in histo.hists_reco
        h5writehist(filename, "hists_reco/"*name, hist)
    end
end

"""

Fill a histogram bin with a given weight and error.

"""
function fill_hist_by_bin!(h::Hist2D, xbin::Int64, ybin::Int64, w::Float64, werr::Float64) # This is a bit of a hacky way to fill the histograms, but it works
    bincounts(h)[xbin, ybin] += w
    sumw2(h)[xbin, ybin] += werr
end

"""

Fill all histograms based on an event.

"""
function fill_all_hists_from_event!(hs::HistogramsOscillations, e::KM3io.ResponseMatrixBin; MC_scaling::Float64=1.)
    W = e.W * MC_scaling
    if hasproperty(e,:WE)
        WE = e.WE * MC_scaling^2
    else
        WE = e.W^2
    end
    if hasproperty(e, :IsCC)
        if Bool(e.IsCC)
            if Particle(e.Pdg).pdgid.value == Particle("nu(e)0").pdgid.value
                fill_hist_by_bin!(hs.hists_true["elec_cc_nu"], e.E_true_bin, e.Ct_true_bin, W, WE)
            elseif Particle(e.Pdg).pdgid.value == Particle("~nu(e)0").pdgid.value
                fill_hist_by_bin!(hs.hists_true["elec_cc_nub"], e.E_true_bin, e.Ct_true_bin, W, WE)
            elseif Particle(e.Pdg).pdgid.value == Particle("nu(mu)0").pdgid.value
                fill_hist_by_bin!(hs.hists_true["muon_cc_nu"], e.E_true_bin, e.Ct_true_bin, W, WE)
            elseif Particle(e.Pdg).pdgid.value == Particle("~nu(mu)0").pdgid.value
                fill_hist_by_bin!(hs.hists_true["muon_cc_nub"], e.E_true_bin, e.Ct_true_bin, W, WE)
            elseif Particle(e.Pdg).pdgid.value == Particle("nu(tau)0").pdgid.value
                fill_hist_by_bin!(hs.hists_true["tau_cc_nu"], e.E_true_bin, e.Ct_true_bin, W, WE)
            elseif Particle(e.Pdg).pdgid.value == Particle("~nu(tau)0").pdgid.value
                fill_hist_by_bin!(hs.hists_true["tau_cc_nub"], e.E_true_bin, e.Ct_true_bin, W, WE)
            end
        else
            if Particle(e.Pdg).pdgid.value == Particle("nu(mu)0").pdgid.value
                fill_hist_by_bin!(hs.hists_true["nc_nu"], e.E_true_bin, e.Ct_true_bin, W, WE)
            elseif Particle(e.Pdg).pdgid.value == Particle("~nu(mu)0").pdgid.value
                fill_hist_by_bin!(hs.hists_true["nc_nub"], e.E_true_bin, e.Ct_true_bin, W, WE)
            end
        end
    end
    if KM3io._getanaclassname(e.AnaClass) == "HighPurityTracks"
        if hasproperty(e, :IsCC)
            fill_hist_by_bin!(hs.hists_true["trueHighPurityTracks"], e.E_true_bin, e.Ct_true_bin, W, WE)
        end
        fill_hist_by_bin!(hs.hists_reco["recoHighPurityTracks"], e.E_reco_bin, e.Ct_reco_bin, W, WE)
    elseif KM3io._getanaclassname(e.AnaClass) == "LowPurityTracks"
        if hasproperty(e, :IsCC)
            fill_hist_by_bin!(hs.hists_true["trueLowPurityTracks"], e.E_true_bin, e.Ct_true_bin, W, WE)
        end
        fill_hist_by_bin!(hs.hists_reco["recoLowPurityTracks"], e.E_reco_bin, e.Ct_reco_bin, W, WE)
    elseif KM3io._getanaclassname(e.AnaClass) == "Showers"
        if hasproperty(e, :IsCC)
            fill_hist_by_bin!(hs.hists_true["trueShowers"], e.E_true_bin, e.Ct_true_bin, W, WE)
        end
        fill_hist_by_bin!(hs.hists_reco["recoShowers"], e.E_reco_bin, e.Ct_reco_bin, W, WE)
    end
    if hasproperty(e, :IsCC)
        fill_hist_by_bin!(hs.hists_true["true"], e.E_true_bin, e.Ct_true_bin, W, WE)
    end
    fill_hist_by_bin!(hs.hists_reco["reco"], e.E_reco_bin, e.Ct_reco_bin, W, WE)

end

"""

Fill histograms with events from an `KM3io.OscOpenDataTree`, optionally applying oscillations and flux weights.

# Inputs
- `hs::HistogramsOscillations`: Histogram structure to fill
- `f::KM3io.OscOpenDataTree`: Struct containing the bins with the weights to fill. 
- `flux_dict::Dict (optional)`: If filling neutrinos from MC, `flux_dict` contains the information of the atmospheric neutrino flux to use to compute the event weights.
- `U0::Matrix{ComplexF64} (optional)`: If filling neutrinos from MC, `U0` corresponds to the precomputed PMNS matrix
- `H0::Vector{ComplexF64} (optional)`: If filling neutrinos from MC, `H0` corresponds to the precomputed hamiltonian.
- `oscillations::Bool (optional)`: Boolean to whether compute the weights using oscillations or not.
- `MC_scaling::Float64 (optional)`: If doing sensitivity studies, this argument allows to scale the MC by a certain value.

"""
function fill_response!(hs::HistogramsOscillations, f::KM3io.OscOpenDataTree,  flux_dict::Union{Dict, Nothing}=nothing, U0::Union{Matrix{ComplexF64}, Nothing}=nothing, H0::Union{Vector{ComplexF64}, Nothing}=nothing; oscillations::Union{Bool, Nothing}=true, MC_scaling::Union{Float64, Nothing}=1.)
    if f.tpath == KM3io.ROOT.TTREE_OSC_OPENDATA_NU
        for e in f
           fill_all_hists_from_event_oscillations_and_flux!(hs, e, flux_dict, U0, H0; oscillations=oscillations, MC_scaling=MC_scaling)
        end
    else
        for e in f
            fill_all_hists_from_event!(hs, e; MC_scaling=MC_scaling)
        end
    end
end


"""

Retrieve a dictionary of neutrino fluxes from the Honda flux model file given as input.

# Inputs
- `flux_path::String`: String of path to input flux file.

# Notes
- If no input is given, default argument is the Honda flux at Frejus site without mountain at solar-min with averaged azimuth.
"""
function get_flux_dict(flux_path::String="")
    if flux_path == ""
        NUFLUX_PATH = split(Base.pathof(NuFlux), "src")[1]
        FLUX_DATA_DIR = joinpath(NUFLUX_PATH, "data")
        flux_path = joinpath(FLUX_DATA_DIR, "frj-nu-20-01-000.d")
    end
    return NuFlux.readfluxfile(flux_path)
end


"""

Compute the PMNS matrix and Hamiltonian for neutrino oscillations for a given input dictionary containing the parameters of neutrino oscillations.

# Inputs
- `nu_params::Dict`: Dictionary of oscillation parameters

# Example input
```julia-repl
NuFitv5 = Dict(
    "dm_21" => 7.42e-5,
    "dm_31" => 2.510e-3,
    "theta_12" => deg2rad(33.45),
    "theta_23" => deg2rad(42.1),
    "theta_13" => deg2rad(8.62),
    "dcp" => deg2rad(230)
)
```

# Outputs
- `U::Matrix{ComplexF64}`: PMNS matrix computed from the input oscillation parameters.
- `H::Vector{ComplexF64}`: Form of the Hamiltonian of propagation.

# Notes
- If no input is given, default parameters are based on NuFit v5.1 results http://www.nu-fit.org/?q=node/238.
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


"""

Compute the weight for an event, considering oscillations and flux.

"""
function osc_weight_computation(E::Float64, zdir::Float64, Pdg::Int16, IsCC::Int16, flux_dict::Dict,U0::Union{Matrix{ComplexF64}, Nothing}, H0::Union{Vector{ComplexF64},Nothing}, oscillations::Bool)
	weight = 0
	isAnti = (Pdg<0)
	path = Neurthino.prempath(acos(zdir), 2, samples=20);
	osc_values = oscprob(U0, H0, E, path, anti = isAnti);
	
    if oscillations
    	for flav in [NUE_PDGID, NUMU_PDGID]
    		nuin = flav < NUE_PDGID+1 ? 1 : 2
    		if abs(Pdg)==NUE_PDGID
    			nuout=1
    		elseif abs(Pdg)==NUMU_PDGID
    			nuout=2
    		elseif abs(Pdg)==NUTAU_PDGID
    			nuout=3
    		end
    		flux_value = NuFlux.flux(flux_dict[flav * sign(Pdg)], E, zdir; interpol=true, interpol_method="linear", interp_logflux=true)
    		if (IsCC > 0)
    			osc_prob_cc = osc_values[1,1,nuin, nuout]
    			weight += flux_value*osc_prob_cc
    		else
    			weight += flux_value
    		end
        end
    else
        if abs(Pdg)==NUTAU_PDGID
            weight = 0
        else
    		if (IsCC > 0)
                flux_value = NuFlux.flux(flux_dict[Pdg], E, zdir; interpol=true, interpol_method="linear", interp_logflux=true)
                weight += flux_value
            else
                for flav in [NUE_PDGID, NUMU_PDGID]
                    flux_value = NuFlux.flux(flux_dict[flav * sign(Pdg)], E, zdir; interpol=true, interpol_method="linear", interp_logflux=true)
                    weight += flux_value
                end
            end
        end
    end
	return weight
end

"""

Fill histograms for an event, applying oscillations and flux weights.

"""
function fill_all_hists_from_event_oscillations_and_flux!(hs::HistogramsOscillations, e::KM3io.ResponseMatrixBin, flux_dict::Dict, U0::Union{Matrix{ComplexF64}, Nothing}=nothing, H0::Union{Vector{ComplexF64}, Nothing}=nothing; oscillations::Bool=true, MC_scaling::Float64=1.)

    if oscillations && (U0 == nothing || H0 == nothing)
        error("Oscillations are enabled, but no PMNS matrix or Hamiltonian was provided.")
    end
	
	weight = osc_weight_computation(e.E_true_bin_center, e.Ct_true_bin_center, e.Pdg, e.IsCC, flux_dict, U0, H0, oscillations)
	

	new_W = e.W * weight * MC_scaling
	new_WE = e.WE * weight^2 * MC_scaling^2
    new_e = KM3io.ResponseMatrixBinNeutrinos(e.E_reco_bin, e.Ct_reco_bin,  e.E_reco_bin_center, e.Ct_reco_bin_center,  e.E_true_bin, e.Ct_true_bin,  e.E_true_bin_center, e.Ct_true_bin_center,  e.Pdg, e.IsCC, e.AnaClass, new_W, new_WE)

    fill_all_hists_from_event!(hs, new_e)

end

"""

Create histograms needed for detector response for oscillations analysis from either a ROOT file or a JSON file.

# Inputs
- `fpath::String`: String to ROOT file or JSON file containing the definition of the true and reco axes for the histograms.

# Outputs
- `HistogramsOscillations`: Structure containing two dictionaries (`hists_true` and `hists_reco`), each structure will contain the empty histograms for each case to fill.

"""
function create_histograms(fpath::String)
    if endswith(fpath, ".root")
        # Handle ROOT file
        fhist = UnROOT.ROOTFile(fpath)
        xbins_true = fhist["hbinstrue"][:fXaxis_fXbins ]
        ybins_true = fhist["hbinstrue"][:fYaxis_fXbins ]
        xbins_reco = fhist["hbinsreco"][:fXaxis_fXbins ]
        ybins_reco = fhist["hbinsreco"][:fYaxis_fXbins ]
        bins_true = HistogramDefinitions(xbins_true, ybins_true)
        bins_reco = HistogramDefinitions(xbins_reco, ybins_reco)
        return HistogramsOscillations(bins_true, bins_reco)
    elseif endswith(fpath, ".json")
        # Handle JSON file
        hist_edges = JSON.parsefile(fpath)
        xbins_true = Float64.(hist_edges["E_true_bins"])
        ybins_true = Float64.(hist_edges["cosT_true_bins"])
        xbins_reco = Float64.(hist_edges["E_reco_bins"])
        ybins_reco = Float64.(hist_edges["cosT_reco_bins"])
        bins_true = HistogramDefinitions(xbins_true, ybins_true)
        bins_reco = HistogramDefinitions(xbins_reco, ybins_reco)
        return HistogramsOscillations(bins_true, bins_reco)
    else
        error("Unsupported file type. Expected a .root or .json file.")
    end
end

"""

Fill an HDF5 file with datasets for neutrino, muon, and data events.

"""
function fill_HDF5_file!(h5file::H5File, f::KM3io.OscOpenDataTree, filetype::String="neutrinos")
    for e in f
        if filetype=="neutrinos"
            if Bool(e.IsCC)
                if Particle(e.Pdg).pdgid.value == Particle("nu(e)0").pdgid.value
                    push!(h5file._datasets["elec_cc_nu"], e)
                elseif Particle(e.Pdg).pdgid.value == Particle("~nu(e)0").pdgid.value
                    push!(h5file._datasets["elec_cc_nub"], e)
                elseif Particle(e.Pdg).pdgid.value == Particle("nu(mu)0").pdgid.value
                    push!(h5file._datasets["muon_cc_nu"], e)
                elseif Particle(e.Pdg).pdgid.value == Particle("~nu(mu)0").pdgid.value
                    push!(h5file._datasets["muon_cc_nub"], e)
                elseif Particle(e.Pdg).pdgid.value == Particle("nu(tau)0").pdgid.value
                    push!(h5file._datasets["tau_cc_nu"], e)
                elseif Particle(e.Pdg).pdgid.value == Particle("~nu(tau)0").pdgid.value
                    push!(h5file._datasets["tau_cc_nub"], e)
                end
            else
                if Particle(e.Pdg).pdgid.value == Particle("nu(mu)0").pdgid.value
                    push!(h5file._datasets["nc_nu"], e)
                elseif Particle(e.Pdg).pdgid.value == Particle("~nu(mu)0").pdgid.value
                    push!(h5file._datasets["nc_nub"], e)
                end
            end
                	
        elseif filetype=="atm_muons"
            push!(h5file._datasets["atm_muons"], e)
        elseif filetype=="data"
            push!(h5file._datasets["data"], e)

        end
    end
   
end


"""

Build an HDF5 file with datasets for neutrino, muon, and data events.

"""
function build_HDF5_file(filename::String="data_MC.h5")
    if isfile(filename)
        @warn "File $filename already exists. It will be overwritten."
        rm(filename)
    end
    fh5 = KM3io.H5File(filename, "w")
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
        KM3io.create_dataset(fh5, pid, KM3io.ResponseMatrixBinNeutrinos)
    end
    KM3io.create_dataset(fh5, "atm_muons", KM3io.ResponseMatrixBinMuons)
    KM3io.create_dataset(fh5, "data", KM3io.ResponseMatrixBinData)
    return fh5
end
