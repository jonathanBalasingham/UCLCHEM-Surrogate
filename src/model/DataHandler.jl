using DelimitedFiles, Serialization

function _extract_physical_parameters(fp::String)
    fp |> 
        x -> replace(x, "CVODE_" => "") |>
        x -> replace(x, ".csv" => "") |>
        x -> split(x, '_') .|> 
        x -> parse(Float64, x)
end

function _extract_esn_parameters(fp::String)
    fp |> 
        x -> replace(x, "W_species_" => "") |>
        x -> replace(x, "W_out_species_" => "") |>
        x -> replace(x, "W_in_species_" => "") |>
        x -> replace(x, "W_time_" => "") |>
        x -> replace(x, "W_out_time_" => "") |>
        x -> replace(x, "W_in_time_" => "") |>
        x -> replace(x, ".csv" => "") |>
        x -> split(x, '_') .|> 
        x -> parse(Float64, x)
end

function _rdaep(path::String)
    params = _extract_physical_parameters(basename(path))
    data = readdlm(path, ',', header=false)
    (params, data)
end

function read_data_and_extract_parameters(datapath::String)
    fullpaths = filter(x->endswith(x, ".csv") && startswith(basename(x), "CVODE_"), readdir(datapath, join=true))
    @info "Found $(length(fullpaths)) files in datapath directory. Reading data and extracting parameters from filename"
    full_set = _rdaep.(fullpaths)
    #= At this point we have an Array{Tuple{String, Matrix}} =#
    params = map(x -> x[begin], full_set)
    data = map(x -> x[end], full_set)
    (params, data)
end


function read_data(datapath::String, predicate::Function)
    fullpaths = filter(x->predicate(basename(x)), readdir(datapath, join=true))
    readdlm.(fullpaths, ',', header=false)
end


function deserialize_model(model_folder_path::String)
    filepaths = readdir(model_folder_path, join=true)
    reservoir_filepath = filter(x->startswith(x, "W_species_") || startswith(x, "W_time"), filepaths)
    W = deserialize(reservoir_filepath)
    input_weight_filepath = filter(x->startswith(x, "W_in_species_") || startswith(x, "W_time"), filepaths)
    W_in = deserialize(reservoir_filepath)
    output_weight_filepath = filter(x->startswith(x, "W_out_species_") || startswith(x, "W_time"), filepaths)
    output_weights = deserialize.(output_weight_filepath)
    physical_parameter_set = _extract_esn_parameters.(output_weight_filepath)
    return (W_in, W, (physical_parameter_set, output_weights))
end

function deserialize_output_weights(model_folder_path::String, type=:species)
    filepaths = readdir(model_folder_path, join=true)

    if type == :species
        output_weight_filepath = filter(x->startswith(basename(x), "W_out_species_"), filepaths)
        output_weights = deserialize.(output_weight_filepath)
        physical_parameter_set = _extract_esn_parameters.(basename.(output_weight_filepath))
        return (physical_parameter_set, output_weights)
    elseif type==:time
        output_weight_filepath = filter(x->startswith(basename(x), "W_out_time"), filepaths)
        output_weights = deserialize.(output_weight_filepath)
        physical_parameter_set = _extract_esn_parameters.(basename.(output_weight_filepath))
        return (physical_parameter_set, output_weights)
    else
        @error "Type must be either :species or :time"
    end
end

function deserialize_reservoir(model_folder_path::String, type=:species)
    filepaths = readdir(model_folder_path, join=true)

    if type == :species
        reservoir_filepath = filter(x->startswith(basename(x), "W_species"), filepaths)[begin]
        W = deserialize(reservoir_filepath)
        return W
    elseif type==:time
        reservoir_filepath = filter(x->startswith(basename(x), "W_time"), filepaths)[begin]
        W = deserialize(reservoir_filepath)
        return W
    else
        @error "Type must be either :species or :time"
    end
end

function deserialize_input_weights(model_folder_path::String, type=:species)
    filepaths = readdir(model_folder_path, join=true)

    if type == :species
        input_weights_filepath = filter(x->startswith(basename(x), "W_in_species"), filepaths)[begin]
        W_in = deserialize(input_weights_filepath)
        return W_in
    elseif type==:time
        input_weights_filepath = filter(x->startswith(basename(x), "W_in_time"), filepaths)[begin]
        W_in = deserialize(input_weights_filepath)
        return W_in
    else
        @error "Type must be either :species or :time"
    end
end