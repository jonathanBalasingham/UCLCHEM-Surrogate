using DrWatson
@quickactivate "UCLCHEM Surrogate"
using Surrogates, ReservoirComputing

function _extract_parameters(fp::String)
    fp |> 
        x -> replace(x, "CVODE_" => "") |>
        x -> replace(x, ".csv" => "") |>
        x -> split(x, '_') .|> 
        x -> parse(Float64, x)
end

function create_surrogate(time_alpha::Float64,
                          species_alpha::Float64,
                          time_beta::Float64,
                          species_beta::Float64;
                          activation=tanh,
                          nla_type=NLADefault(),
                          extended_states=true,
                          W_in_time=readdlm(projectdir("_research", "W_in_time_1.csv"), ',', header=true),
                          W_in_species=readdlm(projectdir("_research", "W_in_species_1.csv"), ',', header=true),
                          W_time=readdlm(projectdir("_research", "W_time_1.csv"), ',', header=true),
                          W_species=readdlm(projectdir("_research", "W_species_1.csv"), ',', header=true),
                          datapath=datadir("sims", "Adaptive"),
                          transform=x->log10.(x .+ 10e-30),
                          rates_set_lower_bound = [1e-17, 0.5, 10, 0.5, 2., 1e2],
                          rates_set_upper_bound = [1., 0.5, 100, 1.5, 10., 1e4])

    resulting_weights_time = []
    resulting_weights_species = []
    parameter_set = Tuple{7, Float64}[]

    for path in readdir(datapath, join=true)
        params = _extract_parameters(basename(path))
        push!(parameter_set, params)
        data = readdlm(path, ',', header=true)
        time_data = data[1:1, :] |> transform
        species_data = data[2:end, :] |> transform
        time_esn = ESN(W_time, time_data, W_in_time, activation=activation, nla_type=nla_type, extended_states=extended_states)
        W_out_time = ESNtrain(time_esn, time_beta)
        species_esn = ESN(W_species, species_data, W_in_species, activation=activation, nla_type=nla_type, extended_states=extended_states)
        W_out_species = ESNtrain(species_esn, species_beta)
        flattened_W_out_time = reshape(W_out_time, :, 1)
        push!(resulting_weights_time, flattened_W_out_time)
        flattened_W_out_species = reshape(W_out_species, :, 1)
        push!(resulting_weights_species, flattened_W_out_species)

    end

    weight_surrogate_species = RadialBasis(parameter_set, resulting_weights_species, rates_set_lower_bound, rates_set_upper_bound)
    weight_surrogate_time = RadialBasis(parameter_set, resulting_weights_time, rates_set_lower_bound, rates_set_upper_bound)
    return weight_surrogate_time, weight_surrogate_species
end

