using KM3OpenOsc
using KM3io
import UnROOT
using KM3NeTTestData
using Test
using FHist

const OSCFILE = datapath("oscillations", "ORCA6_433kt-y_opendata_v0.5_testdata.root")
const BINDEF = datapath("oscillations", "bins_433kt-y_v0.4.json")

@testset "Oscillations open data files" begin
    f = OSCFile(OSCFILE)
    nu = f.osc_opendata_nu
    data = f.osc_opendata_data
    muons = f.osc_opendata_muons
    @test 59360 == length(nu)
    @test 1 == nu[1].AnaClass
    @test 1 == nu[1].Ct_reco_bin
    @test 3 == nu[1].Ct_true_bin
    @test 11 == nu[1].E_reco_bin
    @test 30 == nu[1].E_true_bin
    @test -12 == nu[1].Pdg
    @test 1 == nu[1].IsCC
    @test isapprox(nu[1].W, 725.8889579721612)
    @test isapprox(nu[1].WE, 99699.08425875357)

    @test 92 == length(data)
    @test 1 == data[1].AnaClass
    @test 4 == data[1].Ct_reco_bin
    @test 1 == data[1].E_reco_bin
    @test isapprox(data[1].W, 3.0)

    @test 85 == length(muons)
    @test 1 == muons[1].AnaClass
    @test 7 == muons[1].Ct_reco_bin
    @test 1 == muons[1].E_reco_bin
    @test isapprox(muons[1].W, 0.08825455071391969)
    @test isapprox(muons[1].WE, 0.0009736083957537474)

    hn = create_histograms(BINDEF)
    hd = create_histograms(BINDEF)
    hm = create_histograms(BINDEF)
    @test hn isa HistogramsOscillations
    @test 300 == length(hn.hists_reco["recoShowers"].bincounts)
    @test 3200 == length(hn.hists_true["trueShowers"].bincounts)
    @test hasproperty(hn, :hists_true)
    @test hasproperty(hn, :hists_reco)

    NUFLUX_PATH = split(Base.pathof(NuFlux), "src")[1]
    FLUX_DATA_DIR = joinpath(NUFLUX_PATH, "data")
    flux_path = joinpath(FLUX_DATA_DIR, "frj-ally-20-12-solmin.d")
    flux_dict = get_flux_dict(flux_path)
    @test haskey(flux_dict, -12)
    @test haskey(flux_dict, 14)

    BF = Dict("dm_21" => 7.42e-5,
                   "dm_31" => 2.18e-3,
                   "theta_12" => deg2rad(33.45),
                   "theta_23" => deg2rad(45.57299599919429),
                   "theta_13" => deg2rad(8.62),
                   "dcp" => deg2rad(230))

    U,H = get_oscillation_matrices(BF)

    @test true == isa(U, Matrix)
    @test true == isa(H, Vector)

    fill_response!(hn, nu, flux_dict, U, H; oscillations=true)

    @test isapprox(integral(hd.hists_reco["recoShowers"]), 206.89133463351368)

    hdf5_filename = "test_data.h5"
    export_histograms_hdf5(hn, hdf5_filename)
    @test isfile(hdf5_filename)
    rm(filename)

    fh5 = build_HDF5_file(hdf5_filename)
    fill_HDF5_file!(fh5, nu, "neutrinos")
    close(fh5)
    @test isfile(hdf5_filename)
    rm(hdf5_filename)



    close(f)
end

