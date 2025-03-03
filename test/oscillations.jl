using KM3OpenOsc
using KM3io
import UnROOT
using KM3NeTTestData
using Test
using FHist

const OSCFILE = datapath("oscillations", "ORCA6_433kt-y_opendata_v0.4_testdata.root")
const BINDEF = datapath("oscillations", "bins_433kt-y_v0.4.json")

@testset "Oscillations open data files" begin
    f = OSCFile(OSCFILE)
    nu = f.osc_opendata_nu
    data = f.osc_opendata_data
    muons = f.osc_opendata_muons
    @test 59301 == length(nu)
    @test 1 == nu[1].AnaClass
    @test 1 == nu[1].Ct_reco_bin
    @test 18 == nu[1].Ct_true_bin
    @test 10 == nu[1].E_reco_bin
    @test 30 == nu[1].E_true_bin
    @test -12 == nu[1].Flav
    @test 1 == nu[1].IsCC
    @test isapprox(nu[1].W, 52.25311519561337)
    @test isapprox(nu[1].Werr, 2730.388047646041)

    @test 106 == length(data)
    @test 1 == data[1].AnaClass
    @test 6 == data[1].Ct_reco_bin
    @test 2 == data[1].E_reco_bin
    @test isapprox(data[1].W, 2.0)

    @test 99 == length(muons)
    @test 1 == muons[1].AnaClass
    @test 4 == muons[1].Ct_reco_bin
    @test 1 == muons[1].E_reco_bin
    @test isapprox(data[1].W, 2.0)

    hn = create_histograms(BINDEF)
    hd = create_histograms(BINDEF)
    hm = create_histograms(BINDEF)
    @test true == hasproperty(hn, :hists_true)
    @test true == hasproperty(hn, :hists_reco)

    flux_dict = get_flux_dict()
    @test true == haskey(flux_dict, -12)
    @test true == haskey(flux_dict, 14)

    U,H = get_oscillation_matrices()
    @test true == isa(U, Matrix)
    @test true == isa(H, Vector)

    #fill_response!(hn, nu, flux_dict, U, H; oscillations=true, livetime=1.39)
    #fill_response!(hd, data)
    #fill_response!(hm, muons; livetime=1.39)

    #@test isapprox(integral(hn.hists_true["true"]), 557.9265650274699)
    #@test isapprox(integral(hd.hists_reco["recoShowers"]), 235.0)
    #@test isapprox(integral(hm.hists_reco["recoLowPurityTracks"]), 17.39294089847084)


    close(f)
end

