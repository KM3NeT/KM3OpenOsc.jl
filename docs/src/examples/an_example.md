# An example

The following function determines the meaning of life.

```@example usage
using KM3OpenOsc
using KM3io
using KM3NeTTestData

OSCFILE = KM3NeTTestData.datapath("oscillations", "ORCA6_433kt-y_opendata_v0.4_testdata.root")
BINDEF = KM3NeTTestData.datapath("oscillations", "bins_433kt-y_v0.4.json")

f = KM3io.OSCFile(OSCFILE)
nu = f.osc_opendata_nu
data = f.osc_opendata_data

hn = create_histograms(BINDEF)
hd = create_histograms(BINDEF)

flux_dict = get_flux_dict()
U,H = get_oscillation_matrices()

fill_response!(hn, nu, flux_dict, U, H; oscillations=true, livetime=1.39) 
fill_response!(hd, data)
export_histograms_hdf5(hn, "neutrino_histograms_from_testdata.h5") 
build_HDF5_file("responses_to_file.h5")
fill_HDF5_file("responses_to_file.h5", nu, hn, "neutrinos")
```
