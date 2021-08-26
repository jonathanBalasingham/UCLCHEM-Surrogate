#=

This is a very simple example of how to create 
and predict using the ESNSurrogate Module. This
example uses relatively few example from the 
simplified network. I've noted in the README and 
Main.jl a problem with solving a network repeatedly
The small network has a higher limit for this. I've 
gotten 300+ samples without issue, but I cannot 
determine for certain when it will crash.

=#
using DrWatson
@quickactivate "UCLCHEM Surrogate"

include(srcdir("EchoStateNetwork.jl"))
include(srcdir("ESNSurrogate.jl"))
include(srcdir("Scoring.jl"))
include(srcdir("Transform.jl"))

using DifferentialEquations, Plots, Sundials, Surrogates, Serialization
include.(srcdir.(["GasPhaseNetwork.jl", "CVODESolve.jl", "Visualize.jl"]))
rfp, icfp, sfp = map(x -> datadir("exp_raw", x), ["reactions_small.csv", "initcond0.csv", "species.csv"])
include(srcdir("Constants.jl"))

# Samples we want to include in our interpolation
rates_samples = sample(20, true_dark_cloud_lower, true_dark_cloud_upper, SobolSample())

# DataStructure to store our ESN weights in
weight_dict = Dict()
timepoints = deserialize(projectdir("models", "full_timepoints"))

# ESN we want to make a surrogate from
esn = ESN.EchoStateNetwork{Float64, ESN.EchoStateReservoir{Float64}}(23, 300, 23);
beta = 1e-2

# Run through each sample, solve, train, then store in the Dict
for rates in rates_samples
    ground_truth_solution = direct_solve(rfp, icfp, [rates...], timepoints)
    X = ground_truth_solution[2:end, begin:end-1]
    y = ground_truth_solution[2:end, begin+1:end]
    ESN.train!(esn, [X], [y], beta)
    weight_dict[r([rates...])] = [reshape(esn.output_layer.weight, :, 1)...]
end

# Make the interpolation
interpolation = RadialBasis(r.(rates_samples .|> x->[x...]), [values(weight_dict)...], true_dark_cloud_lower, true_dark_cloud_upper)

# Make the Surrogate
esns = EchoStateNetworkSurrogate.ESNSurrogate(esn, interpolation, timepoints)

# Create a set of test rates
test_sample = sample(1, true_dark_cloud_lower, true_dark_cloud_upper, SobolSample())[begin]
solution_of_test = direct_solve(rfp, icfp, [test_sample...], esns.timepoints)

# Make a prediction on the test rates.
warmup = solution_of_test[:, begin:10]
EchoStateNetworkSurrogate.predict(esns, r([test_sample...]), warmup[begin+1:end, :])