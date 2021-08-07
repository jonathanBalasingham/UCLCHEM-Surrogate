using DrWatson
@quickactivate "UCLCHEM Surrogate"

using DelimitedFiles, ReservoirComputing, Serialization, DataFrames

datapath = datadir("sims", "dark_cloud_low_tol")
paths = readdir(datapath, join=true)

W = deserialize("models/reservoir")
W_in = deserialize("models/input_weights")
settings = deserialize("models/settings")

include(srcdir("model", "DataHandler.jl"))

size_threshold = 5e8 # 500MB
makepath(x) = projectdir("models", x)
data_prefix = "CVODE_adaptive_"
time_threshold = 24*3600*365*10^7 - 24*3600*365*10^6
viable_interpolation_points = 0

for path in paths
    println(path)
    if stat(path).size < size_threshold
        params, data = _rdaep(path, prefix=data_prefix)
        if data[1, end] < time_threshold
            @info "Removing this file. Ending simulation time is: $(data[1,end])"
            rm(path)
            continue
        end
        weights_filename = path |> basename |>
                            x -> replace(x, data_prefix => "W_out_species_") |>
                            x -> replace(x, ".csv" => "")
        full = makepath(weights_filename)
        if !isfile(full)
            try
                data = data[2:end,:] .|> x->log10.(x .+1e-30)
                esn = ESN(W, data, W_in, alpha=settings.alpha, activation=tanh, nla_type=NLADefault(), extended_states=true)
                W_out = ESNtrain(esn, settings.beta)
                serialize(full, W_out)
                viable_interpolation_points += 1
                if viable_interpolation_points % 10 == 0
                   @info "Interpolation points: $viable_interpolation_points" 
                end
            catch DomainError
                @info "Caught a domain error, species concentration went too far into the negatives"
            end
        else
            @info "Found weights for $(basename(full)). Skipping."
        end
    end
end

