const NUE_PDGID = Particle("nu(e)0").pdgid.value
const ANUE_PDGID = Particle("~nu(e)0").pdgid.value
const NUMU_PDGID = Particle("nu(mu)0").pdgid.value
const ANUMU_PDGID = Particle("~nu(mu)0").pdgid.value
const NUTAU_PDGID = Particle("nu(tau)0").pdgid.value
const ANUTAU_PDGID = Particle("~nu(tau)0").pdgid.value

"""
    HistogramDefinitions

A structure defining the binning for histograms.

# Fields
- `xbins::Union{Int64,Vector{Float64}}`: Bin edges or number of bins for the x-axis.
- `ybins::Union{Int64,Vector{Float64},Nothing}`: Bin edges or number of bins for the y-axis.
"""
struct HistogramDefinitions
    xbins::Union{Int64,Vector{Float64}}
    ybins::Union{Int64,Vector{Float64},Nothing}
end

"""
    HistogramsOscillations

A structure containing histograms for oscillation analysis.

# Fields
- `hists_true::Dict{String,Hist2D}`: Histograms for true values.
- `hists_reco::Dict{String,Hist2D}`: Histograms for reconstructed values.
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

"""
    _build_hist(xbins::Union{Int64,Vector{Float64}}, ybins::Union{Int64,Vector{Float64}})

Build a 2D histogram with the given bin edges.

# Arguments
- `xbins::Union{Int64,Vector{Float64}}`: Bin edges or number of bins for the x-axis.
- `ybins::Union{Int64,Vector{Float64}}`: Bin edges or number of bins for the y-axis.

# Returns
- `Hist2D`: A 2D histogram.
"""
function _build_hist(xbins::Union{Int64,Vector{Float64}}, ybins::Union{Int64,Vector{Float64}})
    return Hist2D(; binedges=(xbins, ybins))
end


"""
    export_histograms_hdf5(histo::HistogramsOscillations, filename::String)

Export histograms to an HDF5 file.

# Arguments
- `histo::HistogramsOscillations`: The histograms to export.
- `filename::String`: The name of the HDF5 file.
"""
function export_histograms_hdf5(histo::HistogramsOscillations, filename::String)
    for (name, hist) in histo.hists_true
        h5writehist(filename, "hists_true/"*name, hist)
    end
    for (name, hist) in histo.hists_reco
        h5writehist(filename, "hists_reco/"*name, hist)
    end
end

"""
    fill_hist_by_bin!(h::Hist2D, xbin::Int64, ybin::Int64, w::Float64, werr::Float64)

Fill a histogram bin with a given weight and error.

# Arguments
- `h::Hist2D`: The histogram to fill.
- `xbin::Int64`: The x-axis bin index.
- `ybin::Int64`: The y-axis bin index.
- `w::Float64`: The weight to fill.
- `werr::Float64`: The error on the weight.
"""
function fill_hist_by_bin!(h::Hist2D, xbin::Int64, ybin::Int64, w::Float64, werr::Float64) # This is a bit of a hacky way to fill the histograms, but it works
    bincounts(h)[xbin, ybin] += w
    sumw2(h)[xbin, ybin] += werr
end

"""
    fill_all_hists_from_event!(hs::HistogramsOscillations, e::KM3io.ResponseMatrixBin; livetime::Float64=1.)

Fill all histograms based on an event.

# Arguments
- `hs::HistogramsOscillations`: The histograms to fill.
- `e::KM3io.ResponseMatrixBin`: The event to use for filling.
- `livetime::Float64=1.`: The livetime scaling factor.
"""
function fill_all_hists_from_event!(hs::HistogramsOscillations, e::KM3io.ResponseMatrixBin; livetime::Float64=1.)
    W = e.W * livetime
    if hasproperty(e,:Werr)
        Werr = e.Werr * livetime^2
    else
        Werr = e.W^2
    end
    if hasproperty(e, :IsCC)
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
    if KM3io._getanaclassname(e.AnaClass) == "HighPurityTracks"
        if hasproperty(e, :IsCC)
            fill_hist_by_bin!(hs.hists_true["trueHighPurityTracks"], e.E_true_bin, e.Ct_true_bin, W, Werr)
        end
        fill_hist_by_bin!(hs.hists_reco["recoHighPurityTracks"], e.E_reco_bin, e.Ct_reco_bin, W, Werr)
    elseif KM3io._getanaclassname(e.AnaClass) == "LowPurityTracks"
        if hasproperty(e, :IsCC)
            fill_hist_by_bin!(hs.hists_true["trueLowPurityTracks"], e.E_true_bin, e.Ct_true_bin, W, Werr)
        end
        fill_hist_by_bin!(hs.hists_reco["recoLowPurityTracks"], e.E_reco_bin, e.Ct_reco_bin, W, Werr)
    elseif KM3io._getanaclassname(e.AnaClass) == "Showers"
        if hasproperty(e, :IsCC)
            fill_hist_by_bin!(hs.hists_true["trueShowers"], e.E_true_bin, e.Ct_true_bin, W, Werr)
        end
        fill_hist_by_bin!(hs.hists_reco["recoShowers"], e.E_reco_bin, e.Ct_reco_bin, W, Werr)
    end
    if hasproperty(e, :IsCC)
        fill_hist_by_bin!(hs.hists_true["true"], e.E_true_bin, e.Ct_true_bin, W, Werr)
    end
    fill_hist_by_bin!(hs.hists_reco["reco"], e.E_reco_bin, e.Ct_reco_bin, W, Werr)

end

"""
    fill_response!(hs::HistogramsOscillations, f::KM3io.OscOpenDataTree, flux_dict::Union{Dict, Nothing}=nothing, U0::Union{Matrix{ComplexF64}, Nothing}=nothing, H0::Union{Vector{ComplexF64}, Nothing}=nothing; oscillations::Union{Bool, Nothing}=true, livetime::Union{Float64, Nothing}=1.)

Fill histograms with events from an `KM3io.OscOpenDataTree`, optionally applying oscillations and flux weights.

# Arguments
- `hs::HistogramsOscillations`: The histograms to fill.
- `f::KM3io.OscOpenDataTree`: The oscillation open data tree containing events.
- `flux_dict::Union{Dict, Nothing}=nothing`: Dictionary of neutrino fluxes (optional).
- `U0::Union{Matrix{ComplexF64}, Nothing}=nothing`: PMNS matrix for oscillations (optional).
- `H0::Union{Vector{ComplexF64}, Nothing}=nothing`: Hamiltonian for oscillations (optional).
- `oscillations::Union{Bool, Nothing}=true`: Whether to apply oscillation calculations.
- `livetime::Union{Float64, Nothing}=1.`: Livetime scaling factor for event weights.

# Behavior
- If the tree path corresponds to neutrino events (`TTREE_OSC_OPENDATA_NU`), applies oscillations and flux weights.
- Otherwise, fills histograms without oscillations.
"""
function fill_response!(hs::HistogramsOscillations, f::KM3io.OscOpenDataTree,  flux_dict::Union{Dict, Nothing}=nothing, U0::Union{Matrix{ComplexF64}, Nothing}=nothing, H0::Union{Vector{ComplexF64}, Nothing}=nothing; oscillations::Union{Bool, Nothing}=true, livetime::Union{Float64, Nothing}=1.)
    if f.tpath == KM3io.ROOT.TTREE_OSC_OPENDATA_NU
        for e in f
           fill_all_hists_from_event_oscillations_and_flux!(hs, e, flux_dict, U0, H0; oscillations=oscillations, livetime=livetime)
        end
    else
        for e in f
            fill_all_hists_from_event!(hs, e; livetime=livetime)
        end
    end
end


"""
    get_flux_dict()

Retrieve a dictionary of neutrino fluxes from the Honda flux model.

# Returns
- `Dict`: A dictionary mapping PDG IDs to neutrino fluxes.

# Notes
- Uses the Honda flux model stored in the `NuFlux` package.
- Default file: `frj-ally-20-12-solmin.d`.
"""
function get_flux_dict()
    NUFLUX_PATH = split(Base.pathof(NuFlux), "src")[1]
    FLUX_DATA_DIR = joinpath(NUFLUX_PATH, "data")
    return NuFlux.readfluxfile(joinpath(FLUX_DATA_DIR, "frj-ally-20-12-solmin.d"))
end


"""
    get_oscillation_matrices(nu_params::Dict=Dict(...))

Compute the PMNS matrix and Hamiltonian for neutrino oscillations.

# Arguments
- `nu_params::Dict`: Dictionary of neutrino oscillation parameters. Defaults to NuFit values.

# Returns
- `Tuple{Matrix{ComplexF64}, Vector{ComplexF64}}`: The PMNS matrix (`U0`) and Hamiltonian (`H0`).

# Notes
- Default parameters are based on NuFit v5.1 results http://www.nu-fit.org/?q=node/238.
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
    osc_weight_computation(E::Float64, zdir::Float64, Flav::Int16, IsCC::Int16, flux_dict::Dict, U0::Union{Matrix{ComplexF64}, Nothing}, H0::Union{Vector{ComplexF64}, Nothing}, oscillations::Bool)

Compute the weight for an event, considering oscillations and flux.

# Arguments
- `E::Float64`: Neutrino energy.
- `zdir::Float64`: Cosine of the zenith angle.
- `Flav::Int16`: Neutrino flavor (PDG ID).
- `IsCC::Int16`: Flag indicating charged-current interaction.
- `flux_dict::Dict`: Dictionary of neutrino fluxes.
- `U0::Union{Matrix{ComplexF64}, Nothing}`: PMNS matrix.
- `H0::Union{Vector{ComplexF64}, Nothing}`: Hamiltonian.
- `oscillations::Bool`: Whether to apply oscillation calculations.

# Returns
- `Float64`: The computed event weight.

# Notes
- Uses the `Neurthino` package for oscillation probability calculations.
"""
function osc_weight_computation(E::Float64, zdir::Float64, Flav::Int16, IsCC::Int16, flux_dict::Dict,U0::Union{Matrix{ComplexF64}, Nothing}, H0::Union{Vector{ComplexF64},Nothing}, oscillations::Bool)
	weight = 0
	isAnti = (Flav<0)
	path = Neurthino.prempath(acos(zdir), 2, samples=20);
	osc_values = oscprob(U0, H0, E, path, anti = isAnti);
	
    if oscillations
    	for flav in [NUE_PDGID, NUMU_PDGID]
    		nuin = flav < NUE_PDGID+1 ? 1 : 2
    		if abs(Flav)==NUE_PDGID
    			nuout=1
    		elseif abs(Flav)==NUMU_PDGID
    			nuout=2
    		elseif abs(Flav)==NUTAU_PDGID
    			nuout=3
    		end
    		flux_value = NuFlux.flux(flux_dict[flav * sign(Flav)], E, zdir; interpol=true)
    		if (IsCC > 0)
    			osc_prob_cc = osc_values[1,1,nuin, nuout]
    			weight += flux_value*osc_prob_cc
    		else
    			weight += flux_value
    		end
        end
    else
        if abs(Flav)==NUTAU_PDGID
            weight = 0
        else
    		if (IsCC > 0)
                flux_value = NuFlux.flux(flux_dict[Flav], E, zdir; interpol=true)
                weight += flux_value
            else
                for flav in [NUE_PDGID, NUMU_PDGID]
                    flux_value = NuFlux.flux(flux_dict[flav * sign(Flav)], E, zdir; interpol=true)
                    weight += flux_value
                end
            end
        end
    end
	return weight
end

"""
    fill_all_hists_from_event_oscillations_and_flux!(hs::HistogramsOscillations, e::KM3io.ResponseMatrixBin, flux_dict::Dict, U0::Union{Matrix{ComplexF64}, Nothing}=nothing, H0::Union{Vector{ComplexF64}, Nothing}=nothing; oscillations::Bool=true, livetime::Float64=1.)

Fill histograms for an event, applying oscillations and flux weights.

# Arguments
- `hs::HistogramsOscillations`: The histograms to fill.
- `e::KM3io.ResponseMatrixBin`: The event to process.
- `flux_dict::Dict`: Dictionary of neutrino fluxes.
- `U0::Union{Matrix{ComplexF64}, Nothing}=nothing`: PMNS matrix.
- `H0::Union{Vector{ComplexF64}, Nothing}=nothing`: Hamiltonian.
- `oscillations::Bool=true`: Whether to apply oscillation calculations.
- `livetime::Float64=1.`: Livetime scaling factor.

# Throws
- `ErrorException`: If oscillations are enabled but `U0` or `H0` are missing.
"""
function fill_all_hists_from_event_oscillations_and_flux!(hs::HistogramsOscillations, e::KM3io.ResponseMatrixBin, flux_dict::Dict, U0::Union{Matrix{ComplexF64}, Nothing}=nothing, H0::Union{Vector{ComplexF64}, Nothing}=nothing; oscillations::Bool=true, livetime::Float64=1.)

    if oscillations && (U0 == nothing || H0 == nothing)
        error("Oscillations are enabled, but no PMNS matrix or Hamiltonian was provided.")
    end
	
    E = (binedges(hs.hists_true["true"])[1][e.E_true_bin] .* binedges(hs.hists_true["true"])[1][e.E_true_bin+1]).^.5
	zdir = bincenters(hs.hists_true["true"])[2][e.Ct_true_bin]
	weight = osc_weight_computation(E, zdir, e.Flav, e.IsCC, flux_dict, U0, H0, oscillations)
	

	new_W = e.W * weight * livetime
	new_Werr = e.Werr * weight^2 * livetime^2
    new_e = KM3io.ResponseMatrixBinNeutrinos(e.E_reco_bin, e.Ct_reco_bin,  e.E_true_bin, e.Ct_true_bin,  e.Flav, e.IsCC, e.AnaClass, new_W, new_Werr)

    fill_all_hists_from_event!(hs, new_e)

end

"""
    create_histograms(source)

Create histograms from either a ROOT file or a JSON file.

# Arguments
- `source`: The source of the histogram bin definitions. Can be either:
  - `fpath::String`: Path to a ROOT file or JSON file containing histogram bin definitions.

# Returns
- `HistogramsOscillations`: A structure containing the initialized histograms.

# Examples
```julia
# From a ROOT file
histograms = create_histograms("histograms.root")

# From a JSON file
histograms = create_histograms("histograms.json")
```
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
    build_HDF5_file(filename::String="data_MC.h5")

Build an HDF5 file with datasets for neutrino, muon, and data events.

# Arguments
- `filename::String="data_MC.h5"`: The name of the HDF5 file to create.

# Returns
- `H5File`: The created HDF5 file.
"""
function build_HDF5_file(filename::String="data_MC.h5")
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
