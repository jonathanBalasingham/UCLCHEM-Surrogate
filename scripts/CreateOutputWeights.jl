using DrWatson
@quickactivate "UCLCHEM Surrogate"

using DelimitedFiles, ReservoirComputing, Serialization, DataFrames

datapath = datadir("sims", "dark_cloud")
paths = readdir(datapath, join=true)

W = deserialize("models/reservoir")
W_in = deserialize("models/input_weights")
settings = deserialize("models/settings")

include(srcdir("model", "DataHandler.jl"))

size_threshold = 5e8 # 500MB
makepath(x) = projectdir("models", x)
data_prefix = "CVODE_adaptive_"

for path in paths
    if stat(path).size < size_threshold
        params, data = _rdaep(path, prefix=data_prefix)
        weights_filename = path |> basename |>
                            x -> replace(x, data_prefix => "W_out_species_") |>
                            x -> replace(x, ".csv" => "")
        full = makepath(weights_filename)
        if !isfile(full)
            esn = ESN(W, data[2:end,:] .|> x->log10.(x .+1e-30), W_in, alpha=settings.alpha, activation=tanh, nla_type=NLADefault(), extended_states=true)
            W_out = ESNtrain(esn, settings.beta)
            serialize(full, W_out)
        else
            @info "Found weights for $(basename(full)). Skipping."
        end
    end
end

