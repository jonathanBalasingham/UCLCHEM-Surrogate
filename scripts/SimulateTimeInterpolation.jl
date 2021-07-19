using DrWatson
@quickactivate "UCLCHEM Surrogate"

include(srcdir("voltron", "DataHandler.jl"))
include.(srcdir.(["GasPhaseNetwork.jl", "CVODESolve.jl", "Visualize.jl"])) #, "NetworkSurrogate.jl"]))
using Surrogates, ReservoirComputing


fullpath = readdir(projectdir("_research", "output_weights"), join=true)[end]
esn_settings = _extract_esn_parameters(basename(fullpath))

interpolation_data = deserialize_output_weights(fullpath, :time)
flattened_W_out = map(x->reshape(x, :, 1), interpolation_data[2])
W_out_dims = size(interpolation_data[2][begin])
weight_surrogate = RadialBasis(interpolation_data[1], flattened_W_out, rates_set_lower_bound, rates_set_upper_bound)

test_parameters = (rates_set_lower_bound + rates_set_upper_bound) ./ 2
rfp, icfp, sfp = map(x -> datadir("exp_raw", x), ["reactions_final.csv", "initcond4.csv", "species_final.csv"])

tspan = (0., 10^7 * 365. * 24. * 3600.)
pa = Parameters(test_parameters...)
p = formulate_all(rfp, icfp, pa)
@time sol = solve(p, solver=CVODE_BDF)
train_subset = vcat(sol.t', hcat(sol.u...)) .|> x -> log10.(x .+ 1e-30)

W = deserialize_reservoir(fullpath, :time)
W_in = deserialize_input_weights(fullpath, :time)
W_out = reshape(weight_surrogate(test_parameters), W_out_dims)
buffer = 50
esn = ESN(W, train_subset[1:1, begin:buffer], W_in, alpha=esn_settings[begin], extended_states=true)

time_prediction = ESNpredict(esn, size(train_subset, 2)-buffer, W_out)

esn = ESN(W, train_subset[1:1, :], W_in, alpha=esn_settings[begin], extended_states=true)
W_out = ESNtrain(esn, esn_settings[2])
esn = ESN(W, train_subset[1:1, begin:buffer], W_in, alpha=esn_settings[begin], extended_states=true)

time_prediction2 = ESNpredict(esn, size(train_subset, 2)-buffer, W_out)


plot(prediction[1, :], title="", label="Interpolated", legend=:outertopright)
plot!(prediction2[1, :], title="", label="Predicted", legend=:outertopright)
plot!(train_subset[1, buffer+1:end], title="", label="Groud Truth", legend=:outertopright)
savefig(projectdir("plots_interp_time","time.png"))
