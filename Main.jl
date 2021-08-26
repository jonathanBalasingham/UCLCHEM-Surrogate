using DrWatson
@quickactivate "UCLCHEM Surrogate"

"""
Please note, any timing produced by this script will be inaccurate unless run 
more than one time. First time function calls include compilation time. This
is cached and proceeding calls have the accurate timing.
"""

# UCLCHEM Implementation
rfp, icfp = map(x -> datadir("exp_raw", x), ["reactions_final.csv", "initcond0.csv"])
@info "Loading Libraries"
include.(srcdir.(["GasPhaseNetwork.jl", "CVODESolve.jl", "Visualize.jl", "Constants.jl"]))

# ESN Implementation
include(srcdir("EchoStateNetwork.jl"))

# Surrogate Implementation
include(srcdir("ESNSurrogate.jl"))

# All the top-level function we'll use here
include(srcdir("Demo.jl"))

output_path(x) = projectdir("output", x)

esn_surrogate = EchoStateNetworkSurrogate.load_model(projectdir("models", "esn_surrogate"));
desn_surrogate = EchoStateNetworkSurrogate.load_model(projectdir("models", "desn_surrogate"));

using Surrogates
# Sample a single rate set for this demo
# If you change the number of samples, the
# resulting propogation plot will be different
# since only the last sample of the batch is
# taken.
@info "Sampling Rates"
sampled_rates = sample(10, true_dark_cloud_lower, true_dark_cloud_upper, SobolSample())[end];

# Solve the problem at our KNN timepoints and at the Adaptive timepoints
@info "Solving Problem at KNN points"
ground_truth_solution = direct_solve(rfp, icfp, [sampled_rates...], esn_surrogate.timepoints)
@info "Solving Problem at Adaptive points"
adaptive_ground_truth_solution = direct_solve(rfp, icfp, [sampled_rates...])

# take the first 10 points of the solution and preprocess them for prediction
warmup = prepare_warmup(ground_truth_solution)

@info "Predicting using ESN Interpolation"
esn_interpolation_prediction = EchoStateNetworkSurrogate.predict(esn_surrogate, r([sampled_rates...]), warmup) |> x->transform_back(x, u0=ground_truth_solution[2:end, begin])
@info "Predicting using DESN Interpolation"
desn_interpolation_prediction = EchoStateNetworkSurrogate.predict(desn_surrogate, r([sampled_rates...]), warmup) |> x->transform_back(x, u0=ground_truth_solution[2:end, begin])

# Find our best regularization parameter and train the ESN directly 
# Top row is time, not needed
X = ground_truth_solution[2:end, begin:end-1]
y = ground_truth_solution[2:end, begin+1:end]
esn_trained_prediction = train_and_predict(esn_surrogate.esn, X, y, warmup) |> x->transform_back(x, u0=ground_truth_solution[2:end, begin])
desn_trained_prediction = train_and_predict(desn_surrogate.esn, X, y, warmup) |> x->transform_back(x, u0=ground_truth_solution[2:end, begin])

# These can be changed as desired
# Choosing too many can cause Julia to crash i.e. > ~30
plotted_species=["H", "H2", "CO", "H2O", "CH3OH", "#H2O", "#CO", "#CH3OH", "E-", "C2", "C+"]
full_species = formulate_all(rfp, icfp, [sampled_rates...]).species
esn_visualization = visualize(esn_interpolation_prediction, 
                              esn_trained_prediction, 
                              y[:, 9:end],
                              esn_surrogate.timepoints[10:end], 
                              plotted_species=plotted_species,
                              full_species=full_species)

desn_visualization = visualize(desn_interpolation_prediction, 
                              desn_trained_prediction, 
                              y[:, 9:end],
                              desn_surrogate.timepoints[10:end], 
                              plotted_species=plotted_species,
                              full_species=full_species)

savefig(esn_visualization, output_path("esn_simulation_results.png"))
savefig(desn_visualization, output_path("desn_simulation_results.png"))

# Comparison of KNN clustered points and true Adaptive time steps
# Not included in the report but interesting to see.
scatter(esn_surrogate.timepoints[begin+1:end], label="KNN")
scatter!(adaptive_ground_truth_solution[begin, begin+1:end], label="Adaptive", legend=:outertopright)
xlabel!("Step")
ylabel!("time")
yaxis!(:log10)
savefig(output_path("KNN_vs_Adaptive.png"))

@info "Creating Heatmaps.."

# We're only testing one zeta because the number of solutions we can produce is limited
# to roughly 25-30 in a single Julia run.
# This is a strict limit, please dont exceed or Julia will start throwing errors during
# integration. I'm not sure where this bug stems from. If the same problem that throws and
# error is run again in a fresh Julia session it will complete perfectly fine.
zeta = 1e-14
Ts = range(30, stop=300, length=4) |> collect
densities = range(1e2, stop=1e6, length=4) |> collect

esn_mae_heatmap, esn_roc_heatmap, esn_pe_heatmap = create_heatmap(Ts, densities, zeta, esn_surrogate)
savefig(esn_mae_heatmap, output_path("esn_mae_heatmap.png"))
savefig(esn_roc_heatmap, output_path("esn_roc_heatmap.png"))
savefig(esn_pe_heatmap, output_path("esn_pe_heatmap.png"))

desn_mae_heatmap, desn_roc_heatmap, desn_pe_heatmap = create_heatmap(Ts, densities, zeta, desn_surrogate)
savefig(desn_mae_heatmap, output_path("desn_mae_heatmap.png"))
savefig(desn_roc_heatmap, output_path("desn_roc_heatmap.png"))
savefig(desn_pe_heatmap, output_path("desn_pe_heatmap.png"))


