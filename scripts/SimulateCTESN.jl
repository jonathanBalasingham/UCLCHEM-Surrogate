using DrWatson
@quickactivate "UCLCHEM Surrogate"

using ReservoirComputing, DelimitedFiles, Surrogates


datapath(x) = datadir("sims","Adaptive", x)

function extract_parameters(fp::String)
    fp |> 
        x -> replace(x, "CVODE_" => "") |>
        x -> replace(x, ".csv" => "") |>
        x -> split(x, '_') .|> 
        x -> parse(Float64, x)
end

starting_data = readdlm(datapath(readdir(datapath(""))[1]), ',', header=false) 

res_size = 1000
radius = 1.
degree = 500
activation = tanh
alpha = .7
sigma = 0.5
nla_type = NLADefault()
extended_states = false
beta = 0.0005

species_esn = ESN(res_size,
                starting_data[2:end, :],
                degree,
                radius,
                activation = activation,
                alpha = alpha, 
                sigma = sigma, 
                nla_type = nla_type, 
                extended_states = extended_states)

W_out = ESNtrain(species_esn, beta)
output = ESNfitted(species_esn, W_out)

for (i, species) in enumerate(p.species)
    #plot(sol.t ./ (3600 * 24 * 365), output[i, :], title=species, label="Predicted", legend=:outertopright)
    plot(starting_data[1,:] ./ (3600 * 24 * 365), starting_data[i+1, :], title=species, label="Groud Truth", legend=:outertopright)
    plot!(starting_data[1,:] ./ (3600 * 24 * 365), output[i, :], title=species, label="Predicted", legend=:outertopright)
    xticks!([10^0, 10,10^2,10^3,10^4,10^5,10^6])
    xaxis!(:log10)
    xlims!((1., 10^7))
    ylims!((10^-20, 1))
    yaxis!(:log10)
    savefig(projectdir("plots", "$species.png"))
end

W_species = species_esn.W
W_in_species = species_esn.W_in

time_esn = ESN(res_size,
                starting_data[1:1, :],
                degree,
                radius,
                activation = activation,
                alpha = alpha, 
                sigma = sigma, 
                nla_type = nla_type, 
                extended_states = extended_states)

W_time = time_esn.W
W_in_time = time_esn.W_in

transform(data) = data .+ 1. .|> log10
transform_back(data) = 10 .^ data .- 1.

resulting_species_weights = []
resulting_time_weights = []
parameter_samples = NTuple{6, Float64}[]

for filepath in readdir(datapath(""))
    data = readdlm(datapath(filepath), ',', header=false) |> transform
    params = extract_parameters(filepath)

    species_esn = ESN(W_species,
                data[2:end, :],
                W_in_species,
                activation = activation,
                alpha = alpha, 
                nla_type = nla_type, 
                extended_states = extended_states)

    time_esn = ESN(W_time,
                data[1:1, :],
                W_in_time,
                activation = activation,
                alpha = alpha, 
                nla_type = nla_type, 
                extended_states = extended_states)


    @time W_out_species = ESNtrain(species_esn, beta)
    flattened_W_out = reshape(W_out_species, :, 1)
    push!(resulting_species_weights, flattened_W_out)

    @time W_out_time = ESNtrain(time_esn, beta)
    flattened_W_out = reshape(W_out_time, :, 1)
    push!(resulting_time_weights, flattened_W_out)

    push!(parameter_samples, tuple(params...))
end

rates_set_lower_bound = [1e-17, 0.5, 10, 0.5, 2., 1e2]
rates_set_upper_bound = [1., 0.5, 100, 1.5, 10., 1e4]

weight_surrogate_species = RadialBasis(parameter_samples, resulting_species_weights, rates_set_lower_bound, rates_set_upper_bound)
weight_surrogate_time = RadialBasis(parameter_samples, resulting_time_weights, rates_set_lower_bound, rates_set_upper_bound)

test_parameters = rates_set_lower_bound  .+ ((rates_set_upper_bound .- rates_set_lower_bound) .* .5)

include(srcdir("GasPhaseNetwork.jl"))
include(srcdir("CVODESolve.jl"))

rfp = datadir("exp_raw", "reactions_final.csv")
sfp = datadir("exp_raw", "species_final.csv")
icfp = datadir("exp_raw", "initcond4.csv")

pa = Parameters(test_parameters...)
#p = UCLCHEM.formulate(sfp,rfp,icfp,pa,tspan, rate_factor = 1)
p = formulate_all(rfp, icfp, pa)
sol = solve(p, solver=CVODE_BDF)

#train_subset = vcat( log10.(sol.t)' ./ log10(tspan[end]), (hcat(sol.u...) |> Matrix) )
data = vcat(sol.t', hcat(sol.u...)) |> Matrix

species_esn = ESN(W_species,
                data[2:end, :] |> transform,
                W_in_species,
                activation = activation,
                alpha = alpha, 
                nla_type = nla_type, 
                extended_states = extended_states)

time_esn = ESN(W_time,
            data[1:1, :] |> transform,
            W_in_time,
            activation = activation,
            alpha = alpha, 
            nla_type = nla_type, 
            extended_states = extended_states)


@time W_out_species = ESNtrain(species_esn, beta)
@time W_out_time = ESNtrain(time_esn, beta)

test_W_out_species = reshape(weight_surrogate_species(test_parameters), length(p.species), :)
test_W_out_time =  reshape(weight_surrogate_time(test_parameters), 1, :)

@time test_output_species = ESNfitted(species_esn, test_W_out_species)
@time test_output_time = ESNfitted(time_esn, test_W_out_time)

@time output_species = ESNfitted(species_esn, W_out_species)
@time output_time = ESNfitted(time_esn, W_out_time)

using Plots
scatter(data[1,1:10:end], label="Ground Truth", size=(1200,900))
scatter!(transform_back(output_time[1:10:end]), label = "Predicted")
scatter!(transform_back(test_output_time[1:10:end]), label="Interpolated")
savefig("time_pred_beta_$beta.png")

for (i, species) in enumerate(p.species)
    #plot(sol.t ./ (3600 * 24 * 365), output[i, :], title=species, label="Predicted", legend=:outertopright)
    plot(sol.t ./ (3600 * 24 * 365), data[i+1, :], title=species, label="Groud Truth", legend=:outertopright)
    plot!(sol.t ./ (3600 * 24 * 365), test_output_species[i, :] |> transform_back, title=species, label="Interpolated with true t", legend=:outertopright)
    plot!(transform_back(test_output_time)' ./ (3600 * 24 * 365), test_output_species[i, :] |> transform_back, title=species, label="Interpolated with pred t", legend=:outertopright)
    xticks!([10^0, 10,10^2,10^3,10^4,10^5,10^6])
    xaxis!(:log10)
    xlims!((1., 10^7))
    ylims!((10^-20, 1))
    yaxis!(:log10)
    savefig(projectdir("plots", "$species.png"))
end
