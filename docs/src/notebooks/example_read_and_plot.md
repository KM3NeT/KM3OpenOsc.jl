```@raw html
<style>
    #documenter-page table {
        display: table !important;
        margin: 2rem auto !important;
        border-top: 2pt solid rgba(0,0,0,0.2);
        border-bottom: 2pt solid rgba(0,0,0,0.2);
    }

    #documenter-page pre, #documenter-page div {
        margin-top: 1.4rem !important;
        margin-bottom: 1.4rem !important;
    }

    .code-output {
        padding: 0.7rem 0.5rem !important;
    }

    .admonition-body {
        padding: 0em 1.25em !important;
    }
</style>

<!-- PlutoStaticHTML.Begin -->
<!--
    # This information is used for caching.
    [PlutoStaticHTML.State]
    input_sha = "c2712983266fa832ab1f568f2264f46ff0e8c1f9db2d191dc01b27cf9d42ee7a"
    julia_version = "1.11.2"
-->

<div class="markdown"><h1>KM3NeT Open Test Data</h1><p>This notebook demonstrates read and fill histograms from the ORCA 433 kt-y oscillations data release from the KM3NeT neutrino telescope using KM3NeT's tools.</p></div>


```
## Setup and Dependencies
```@raw html
<div class="markdown">
<p>First, we load the necessary packages:</p><ul><li><p><code>KM3io</code>: For reading KM3NeT data formats</p></li><li><p><code>NuFlux</code>: For neutrino flux calculations</p></li><li><p><code>KM3OpenOsc</code>: For neutrino oscillation calculations</p></li><li><p><code>KM3NeTTestData</code>: For accessing test data</p></li><li><p><code>FHist</code>: For histogram manipulation</p></li><li><p><code>CairoMakie</code>: For later plotting</p></li></ul></div>

<pre class='language-julia'><code class='language-julia'>begin
    using KM3io
    using NuFlux
    using KM3OpenOsc
    using KM3NeTTestData
end</code></pre>



```
## Loading Data Files
```@raw html
<div class="markdown">
<p>Here we define paths to two key files:</p><ul><li><p><code>BINDEF</code>: JSON file containing binning definitions for the analysis</p></li><li><p><code>OSCFILE</code>: ROOT file containing data from KM3NeT/ORCA</p></li></ul></div>



```
## Reading the Oscillation Data File
```@raw html
<div class="markdown">
<p>To read the file we use <code>KM3io.OSCFile</code> which is a function that directly identifies the type of file and what is inside</p><p>We load the oscillation data file and extract three data samples:</p><ul><li><p><code>nu</code>: Neutrino Monte Carlo events</p></li><li><p><code>data</code>: Experimental data events</p></li><li><p><code>muons</code>: Atmospheric muon background events</p></li></ul></div>

<pre class='language-julia'><code class='language-julia'>f = KM3io.OSCFile(OSCFILE)</code></pre>
<pre class="code-output documenter-example-output" id="var-hash153331">nothing</pre>

<pre class='language-julia'><code class='language-julia'>begin
    nu = f.osc_opendata_nu
    data = f.osc_opendata_data
    muons = f.osc_opendata_muons
end</code></pre>
<pre class="code-output documenter-example-output" id="var-hash179017">nothing</pre>


```
## Creating Histogram Containers
```@raw html
<div class="markdown">
<p>We initialize empty 2D histogram containers using the binning definition file:</p><ul><li><p><code>hn</code>: For simulated neutrino events</p></li><li><p><code>hd</code>: For real experimental data</p></li><li><p><code>hm</code>: For simulated atmospheric muon events</p></li></ul></div>

<pre class='language-julia'><code class='language-julia'>begin
    hn = create_histograms(BINDEF)
    hd = create_histograms(BINDEF)
    hm = create_histograms(BINDEF); nothing
end</code></pre>
<pre class="code-output documenter-example-output" id="var-hash140462">nothing</pre>

<pre class='language-julia'><code class='language-julia'>fieldnames(typeof(hn))</code></pre>
<pre class="code-output documenter-example-output" id="var-hash181202">nothing</pre>

<pre class='language-julia'><code class='language-julia'>keys(hn.hists_reco)</code></pre>
<pre class="code-output documenter-example-output" id="var-hash383962">nothing</pre>

<pre class='language-julia'><code class='language-julia'>hn.hists_reco["recoShowers"]</code></pre>
<pre class="code-output documenter-example-output" id="var-hash892691">nothing</pre>


```
## Setting Oscillation Parameters
```@raw html
<div class="markdown">
<p>We define the oscillation parameters using the best fit values from the standard oscillations analysis from ORCA 433 kt-y paper JHEP10(2024)206:</p><ul><li><p><code>dm_21</code>, <code>dm_31</code>: Mass-squared differences in eV²</p></li><li><p><code>theta_12</code>, <code>theta_23</code>, <code>theta_13</code>: Mixing angles</p></li><li><p><code>dcp</code>: CP-violating phase</p></li></ul><p>These parameters are used to calculate the oscillation matrices (U and H). It uses <code>Neurthino.jl</code> behind the scenes.</p></div>

<pre class='language-julia'><code class='language-julia'>begin
    BF = Dict("dm_21" =&gt; 7.42e-5,
                       "dm_31" =&gt; 2.18e-3,
                       "theta_12" =&gt; deg2rad(33.45),
                       "theta_23" =&gt; deg2rad(45.57299599919429),
                       "theta_13" =&gt; deg2rad(8.62),
                       "dcp" =&gt; deg2rad(230))

    U,H = get_oscillation_matrices(BF)
end</code></pre>
<pre class="code-output documenter-example-output" id="var-U">(ComplexF64[0.8249422516589652 + 0.0im 0.5449826827686066 + 0.0im -0.09634131252970905 + 0.1148151053225223im; -0.3284406729882627 + 0.06841342277740595im 0.6219810005635141 + 0.04519604930846363im 0.7060759732322109 + 0.0im; 0.449910833254889 + 0.06705856737677285im -0.5586843920183072 + 0.04430098940637267im 0.6920928861894314 + 0.0im], ComplexF64[-0.0007514 + 0.0im, -0.0006772000000000001 + 0.0im, 0.0014286000000000001 + 0.0im])</pre>


```
## Loading Neutrino Flux Model
```@raw html
<div class="markdown">
<p>We load the Honda Frejus site flux model, which provides the atmospheric neutrino flux predictions needed for the analysis. Any Honda formatted table can be read using this function. It uses <code>NuFlux.jl</code> behind the scenes.</p></div>

<pre class='language-julia'><code class='language-julia'>begin
    NUFLUX_PATH = split(Base.pathof(NuFlux), "src")[1]
    FLUX_DATA_DIR = joinpath(NUFLUX_PATH, "data")
    flux_path = joinpath(FLUX_DATA_DIR, "frj-nu-20-01-000.d") # Get flux of Honda Frejus site from NuFlux
    flux_dict = get_flux_dict(flux_path)
end</code></pre>
<pre class="code-output documenter-example-output" id="var-FLUX_DATA_DIR">Dict{Int32, NuFlux.FluxTable} with 4 entries:
  -12 =&gt; FluxTable([-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1  …  0.1,…
  12  =&gt; FluxTable([-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1  …  0.1,…
  14  =&gt; FluxTable([-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1  …  0.1,…
  -14 =&gt; FluxTable([-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1  …  0.1,…</pre>


```
## Filling Histograms with Events
```@raw html
<div class="markdown">
<p>We populate our histogram containers with events:</p><ul><li><p>Neutrinos with flux weighting and oscillation effects applied</p></li><li><p>Atmospheric muons</p></li><li><p>Experimental data</p></li></ul><p>There is an optional argument which is livetime that can be used to scale up or down montecarlo</p></div>

<pre class='language-julia'><code class='language-julia'>begin
    fill_response!(hn, nu, flux_dict, U, H; oscillations=true) # fill neutrinos ,need flux, oscillation parameters
    fill_response!(hm, muons)
    fill_response!(hd, data)
end</code></pre>
<pre class="code-output documenter-example-output" id="var-hash860418">nothing</pre>

<pre class='language-julia'><code class='language-julia'>hn.hists_true["muon_cc_nu"]</code></pre>
<pre class="code-output documenter-example-output" id="var-hash404363">nothing</pre>

<pre class='language-julia'><code class='language-julia'>hn.hists_reco["recoShowers"]</code></pre>
<pre class="code-output documenter-example-output" id="var-hash892691">nothing</pre>


```
## Exporting results to hdf5
```@raw html
<div class="markdown">
<p>You can easily export the filled histograms to HDF5.</p><p>Additionally it is possible to export the full response to a single hdf5 file</p></div>

<pre class='language-julia'><code class='language-julia'>begin
    export_histograms_hdf5(hn, "neutrino_histograms_from_testdata.h5") # You can easily export the filled histograms to hdf5
    h5f = build_HDF5_file("responses_to_file.h5") # Create h5 file with same structure as responses bins 
    fill_HDF5_file!(h5f, nu, "neutrinos") # Completely export the response as a table in an hdf5 file at a given path 
    fill_HDF5_file!(h5f, muons, "muons")
    fill_HDF5_file!(h5f, data, "data") 
    close(h5f)
end</code></pre>
<pre class="code-output documenter-example-output" id="var-hash107358">nothing</pre>


```
## Visualizing the Reconstruction Channels
```@raw html
<div class="markdown">
<p>CAVEAT: These plots are done with a test data sample, meaning it does not contain all events which will render the distributions to be different from the actual analysis.</p><p>This plot shows three different reconstruction channels:</p><ol><li><p>High Purity Tracks: Events classified with track topology with low atmospheric muon contamination</p></li><li><p>Low Purity Tracks: Events classified with track topology with higher atmospheric muon contamination</p></li><li><p>Showers: Events classified with a shower topology</p></li></ol><p>For each channel, we display:</p><ul><li><p>Stacked histograms of neutrino and muon Monte Carlo events</p></li><li><p>Data points with error bars</p></li></ul><p>The x-axis represents the cosine of the zenith angle (cos θ), where -1 means upgoing events (through the Earth) and 0 means horizontal events.</p></div>

<pre class='language-julia'><code class='language-julia'>using CairoMakie, FHist</code></pre>


<pre class='language-julia'><code class='language-julia'>with_theme(ATLASTHEME) do
    reco_types = ["recoHighPurityTracks", "recoLowPurityTracks", "recoShowers"]
     f = Figure(size = (1200, 400))

    for (i, reco_type) in enumerate(reco_types)
        neutrinos = project(hn.hists_reco[reco_type],:y)
        muons = project(hm.hists_reco[reco_type],:y)
        data = project(hd.hists_reco[reco_type],:y)
        ax = Axis(f[1, i], 
                 xlabel = "Cos theta", 
                 ylabel = i == 1 ? "Event count" : "", 
                 title = reco_type)
        p = stackedhist!(ax, [neutrinos, muons]; error_color=Pattern('/'))
        xlims!(ax, (-1, 0))
        scatter!(ax, bincenters(data), bincounts(data), 
                 color = :black, 
                 markersize = 10)
        if i == 3
            labels = ["Neutrinos", "Muons", "Data"]
            elements = [
                [PolyElement(polycolor = p.attributes.color[][j]) for j in 1:2]..., 
                MarkerElement(color = :black, marker = :circle, markersize = 8)
            ]
        end
    end
    f
end</code></pre>
<pre class="code-output documenter-example-output" id="var-hash132083">nothing</pre>


```
## Examining True Event Categories
```@raw html
<div class="markdown">
<p>CAVEAT: These plots are done with a test data sample, meaning it does not contain all events which will render the distributions to be different from the actual analysis.</p><p>This plot shows the distribution of true Monte Carlo event categories:</p><ol><li><p>muon_cc_nu: Charged-current muon neutrino events</p></li><li><p>muon_cc_nub: Charged-current muon antineutrino events</p></li></ol></div>

<pre class='language-julia'><code class='language-julia'>with_theme(ATLASTHEME) do
    reco_types = ["muon_cc_nu", "muon_cc_nub"]
     f = Figure(size = (1200, 400))
    # Loop through each reconstruction type
    for (i, reco_type) in enumerate(reco_types)
        muon_cc_nu = project(hn.hists_true[reco_type],:y)
        muon_cc_nub = project(hm.hists_true[reco_type],:y)
        ax = Axis(f[1, i], 
                 xlabel = "Cos theta", 
                 ylabel = i == 1 ? "Event count" : "", 
                 title = reco_type)
        p = stackedhist!(ax, [muon_cc_nu, muon_cc_nub]; error_color=Pattern('/'))
        xlims!(ax, (-1, 0))
        if i == 3
            labels = ["muon_cc_nu", "muon_cc_nub"]
            elements = [
                [PolyElement(polycolor = p.attributes.color[][j]) for j in 1:2]..., 
                MarkerElement(color = :black, marker = :circle, markersize = 8)
            ]
        end
    end
    f
end</code></pre>
<pre class="code-output documenter-example-output" id="var-hash372987">nothing</pre>
<div class='manifest-versions'>
<p>Built with Julia 1.11.2 and</p>
CairoMakie 0.13.5<br>
FHist 0.11.10<br>
KM3NeTTestData 0.4.21<br>
KM3OpenOsc 0.1.9<br>
KM3io 0.18.6<br>
NuFlux 0.1.6
</div>

<!-- PlutoStaticHTML.End -->
```

