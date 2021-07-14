include("Settings.jl")
include("Scoring.jl")
using ReservoirComputing, Surrogates, DelimitedFiles, Statistics

struct Simulation
    type::String
    training_data_store::String
    training_data_transform::Function
    transform_alias::String
    training_data_transform_back::Function
    settings::RolledSimulationSettings
end

function create_esn(settings::ESNSettings, train_data::T) where {T <: Matrix{Float64}}
    if isnothing(settings.W_path) && isnothing(settings.W_in_path)
        ESN(settings.reservoir_size,
            train_data,
            settings.degree,
            settings.radius,
            activation = settings.activation,
            alpha = settings.alpha, 
            sigma = settings.sigma, 
            nla_type = settings.nla_type, 
            extended_states = settings.extended_states
        )
    elseif isnothing(settings.W_in_path)
        W = readdlm(settings.W_path, ',', header=false)
        ESN(W,
            train_data,
            activation = settings.activation,
            sigma = settings.sigma,
            alpha = settings.alpha,
            nla_type = settings.nla_type,
            extended_states = settings.extended_states
        )
    elseif isnothing(settings.W_path)
        W_in = readdlm(settings.W_in_path, ',', header=false)
        ESN(settings.reservoir_size,
            train_data,
            settings.degree,
            settings.radius,
            W_in;
            activation = settings.activation,
            alpha = settings.alpha,
            nla_type = settings.nla_type,
            extended_states = settings.extended_states)
    else
        W = readdlm(settings.W_path, ',', header=false)
        W_in = readdlm(settings.W_in_path, ',', header=false)

        ESN(W,
            train_data,
            W_in,
            activation = settings.activation,
            alpha = settings.alpha, 
            nla_type = settings.nla_type, 
            extended_states = settings.extended_states
        )
    end
end

function simulate(s::Simulation)
    # create the esn
    # run through all the training train_data
    scores = Array{Float64, 1}[]
    for filepath in readdir(datadir(s.training_data_store))
        current_score = Float64[]
        data = readdlm(datadir(s.training_data_store, filepath), ',', header=false) |> s.training_data_transform
        if s.type == "species"
            data = data[2:end, :]
        elseif s.type == "time"
            data = data[1:1, :]
        end
        esn = create_esn(s.settings.esn_settings, data)
        W_out = ESNtrain(esn, s.settings.esn_settings.beta)
        # train -> predict the fit
        prediction = ESNfitted(esn, W_out, autonomous=true)
        # generate a score for each set of data
        if size(prediction, 1) == 1
            prediction = prediction[begin,:]
            data = data[begin,:]
        end
        mae_score = mae(data, prediction)
        roc_score = roc(data, prediction)
        pmae_score = perc_mae(data, prediction)
        proc_score = perc_roc(data, prediction)
        push!(scores, [mae_score, pmae_score, roc_score, proc_score])
    end
    score_matrix = hcat(scores...)
    mean_scores = mean(score_matrix, dims=2)
    mean_scores
end

adaptivedir(x) = datadir("sims", "Adaptive", x)
staticdir(x) = datadir("sims", "Static", x)

function makesim(alpha,
                sigma,
                beta,
                type;
                datastore=adaptivedir(""),
                transform=x -> log10.(x .+ 1.),
                transform_alias="log10",
                transform_back=x->10. .^x .-1.,
                activation = tanh, 
                nla_type = NLADefault(), 
                extended_states = false,
                W_path=projectdir("_research", "W_default.csv"),
                W_in_path=projectdir("_research", "W_in_default.csv"))

    #=
    reservoir_size::Integer
    radius::Float64
    degree::Integer
    activation::Function
    alpha::Float64
    sigma::Float64
    nla_type
    extended_states::Bool
    beta::Float64
    W_path::String
    W_in_path::String
    =#

    esns = ESNSettings(
        nothing,
        radius,
        nothing,
        activation,
        alpha,
        sigma,
        nla_type,
        extended_states,
        beta,
        W_path,
        W_in_path
    )
    #=
    type::String
    training_data_store::String
    training_data_transform::Function
    transform_alias::String
    training_data_transform_back::Function
    settings::RolledSimulationSettings
    =#
    s = Simulation(type, datastore, transform, transform_alias, transform_back, RolledSimulationSettings(esns, ChemistrySettings()))
    return simulate(s)
end



function makesim(reservoir_size,
                 degree, 
                 alpha,
                 sigma,
                 beta,
                 type,
                 savepath;
                 datastore=adaptivedir(""),
                 transform=x->log10.(x .+ 1),
                 transform_alias="log10",
                 transform_back=x->10 .^x .-1,
                 sample_datapath=readdir(adaptivedir(""), join=true)[1],
                 radius=0.7,
                 activation = tanh, 
                 nla_type = NLADefault(), 
                 extended_states = false,
                 datatore=adaptivedir(""))
    #= 
    We're going to create a new ESN with new reservoir and input weights 
    This will be save according to W_$savepath and W_in_$savepath
    =#
    if type == "time"
        # Assumes the time data is the first row of the data matrix
        sample_data = readdlm(sample_datapath, ',', header=false)[1:1, :]
    else
        # Species concentration account for the rest
        sample_data = readdlm(sample_datapath, ',', header=false)[2:end, :]
    end

    esn = ESN(reservoir_size,
            sample_data,
            degree,
            radius,
            activation = activation,
            alpha = alpha, 
            sigma = sigma, 
            nla_type = nla_type, 
            extended_states = extended_states)
    
    W_path = projectdir("_research", "W_"*savepath*".csv")
    if !isfile(W_path)
        open(W_path, "w") do io
            writedlm(io, esn.W, ',')
        end
    else
        @warn "Reservoir weight file already exists at $W_path"
    end
    W_in_path = projectdir("_research", "W_in_"*savepath*".csv")
    if !isfile(W_in_path)
        open(W_in_path, "w") do io
            writedlm(io, esn.W_in, ',')
        end
    else
        @warn "Input weight file already exists at $W_in_path"
    end
    
    esns = ESNSettings(
        esn.res_size,
        radius,
        degree,
        activation,
        alpha,
        sigma,
        nla_type,
        extended_states,
        beta,
        W_path,
        W_in_path
    )
    

    s = Simulation(type, datastore, transform, transform_alias, transform_back, RolledSimulationSettings(esns, ChemistrySettings()))
    return simulate(s)
end