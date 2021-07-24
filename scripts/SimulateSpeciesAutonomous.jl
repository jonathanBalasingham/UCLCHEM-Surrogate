include(srcdir("Simulation.jl"))
using Surrogates, DataStructures, JLD2, DataFrames

Random.seed!(0)

hyperparameter_lower_bound = [3000, 50, 0.01, 0.1, 1.0]
hyperparameter_upper_bound = [3000, 1500, .9, 0.5, 3.5]

hpnames = ["reservoir_size","degree","alpha", "sigma", "beta", "type", "savepath"]
hp_set = sample(100, hyperparameter_lower_bound, hyperparameter_upper_bound, SobolSample())
radius_set = sample(100, .6, 1.25, SobolSample())

rad_ix = length(radius_set)
for hp in hp_set
    @time begin
        full_params = [hp..., "species", "species_$(hp[1])_$(hp[2] |> round |> Integer)_$(hp[4])_$(radius_set[rad_ix])"]
        full_params[1:2] = full_params[1:2] .|> round .|> Integer
        d = OrderedDict(zip(hpnames, full_params))
        p = copy(d)
        alias = "log"
        mae_score, pmae_score, roc_score, proc_score = makesim(full_params..., 
                                                               datasets=datasets,
                                                               radius=radius_set[rad_ix],
                                                               transform=x->log10.(x .+ 1e-30),
                                                               #transform=x -> x .+ 1e-30,
                                                               transform_alias=alias,
                                                               #transform_back=x->x .^ 10 .- 1e-30)
                                                               transform_back=x-> 10 .^ x .- 1e-30)
        d["transform"] = alias
        d["mae"] = mae_score
        d["pmae"] = pmae_score
        d["roc"] = roc_score
        d["proc"] = proc_score
        d["radius"] = radius_set[rad_ix]
        println("$proc_score , $pmae_score")
        @tagsave(datadir("sims","tuning", "auto_log", savename(Dict(p), "jld2")), Dict(d))
        rad_ix -= 1
    end
end

df = collect_results(datadir("sims", "tuning", "auto_log"))
sort!(df, [:roc, :proc])

top_results = first(df, 1)

data = readdlm(readdir(datadir("sims", "Adaptive"), join=true)[1], ',', header=false)
time = data[1, :]
train_data = data[2:end, :] |> x-> log10.(x .+ 1e-30)
j = 1
buffer=50
for result in eachrow(top_results)
    W = readdlm(projectdir("_research", "W_$(result.savepath).csv"), ',', header=false)
    W_in = readdlm(projectdir("_research", "W_in_$(result.savepath).csv"), ',', header=false)

    esn = ESN(W,
            train_data,
            W_in,
            activation = tanh,
            alpha = result.alpha, 
            nla_type = NLADefault(), 
            extended_states = false)
    W_out = ESNtrain(esn, result.beta)
    esn = ESN(W,
            train_data[:,1:buffer],
            W_in,
            activation = tanh,
            alpha = result.alpha, 
            nla_type = NLADefault(), 
            extended_states = false)
    output = ESNpredict(esn, size(train_data, 2)-buffer, W_out)
    #println(sum.(eachcol(output)))
    for (i,k) in enumerate(eachrow(output))
        plot(output[i, :], title="", label="Predicted", legend=:outertopright)
        plot!(train_data[i, buffer+1:end], title="", label="Groud Truth", legend=:outertopright)
        savefig(projectdir("plots","species$(i)_$(j).png"))
    end
    j += 1
end


buffers = [5,25,50,100,500]
result = top_results[1,:]

for buffer in buffers
    W = readdlm(projectdir("_research", "W_$(result.savepath).csv"), ',', header=false)
    W_in = readdlm(projectdir("_research", "W_in_$(result.savepath).csv"), ',', header=false)

    esn = ESN(W,
            train_data,
            W_in,
            activation = tanh,
            alpha = result.alpha, 
            nla_type = NLADefault(), 
            extended_states = false)
    W_out = ESNtrain(esn, result.beta)
    esn = ESN(W,
            train_data[:,1:buffer],
            W_in,
            activation = tanh,
            alpha = result.alpha, 
            nla_type = NLADefault(), 
            extended_states = false)
    output = ESNpredict(esn, size(train_data, 2)-buffer, W_out)
    #println(sum.(eachcol(output)))
    for (i,k) in enumerate(eachrow(output))
        plot(output[i, :], title="", label="Predicted", legend=:outertopright)
        plot!(train_data[i, buffer:end], title="", label="Groud Truth", legend=:outertopright)
        savefig(projectdir("plots","species$(i)_$(buffer).png"))
    end
end

dataset_paths = readdir(datadir("sims", "Adaptive"), join=true)
#= Write the output weights to a files =#
for result in eachrow(top_results)
    path(x) = projectdir("_research", "output_weights", "$(result.alpha)_$(result.beta)_$(result.sigma)_$(result.reservoir_size)_$(result.degree)_$(result.radius)", x)
    mkdir(path(""))
    W = readdlm(projectdir("_research", "W_$(result.savepath).csv"), ',', header=false)
    W_in = readdlm(projectdir("_research", "W_in_$(result.savepath).csv"), ',', header=false)
    #= open(path * "/W.csv", "w") do io
        writedlm(io, W, ',')
    end
    open(path * "/W_in.csv", "w") do io
        writedlm(io, W_in, ',')
    end =#
    for ds_path in dataset_paths
        train_data = readdlm(ds_path, ',', header=false)[2:end,:] |> x -> log10.(x .+ 1e-30)
        weights_filename = ds_path |> basename |>
                x -> replace(x, "CVODE_" => "W_out_species_") |>
                x -> replace(x, ".csv" => "")
        esn = ESN(W,
                train_data,
                W_in,
                activation = tanh,
                alpha = result.alpha, 
                nla_type = NLADefault(), 
                extended_states = true)
        W_out = ESNtrain(esn, result.beta)
        serialize(path(weights_filename), W_out)
    end
    serialize(path("W_species"), W)
    serialize(path("W_in_species"), W_in)
end