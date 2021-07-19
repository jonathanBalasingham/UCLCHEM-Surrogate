include(srcdir("Simulation.jl"))
using Surrogates, DataStructures, JLD2, DataFrames

hyperparameter_lower_bound = [500, 10, 0.3, 0.0001, 0.000000001]
hyperparameter_upper_bound = [500, 150, .6, 1.0, 0.00001]

hpnames = ["reservoir_size","degree","alpha", "sigma", "beta", "type", "savepath"]
hp_set = sample(1000, hyperparameter_lower_bound, hyperparameter_upper_bound, SobolSample())
radius_set = sample(1000, 0.7, 1.2, SobolSample())

rad_ix = length(radius_set)
for hp in hp_set
    @time begin
        hp = hp .|> x -> round(x, digits=6)
        full_params = [hp..., "time", "time_$(hp[2] |> round |> Integer)_$(hp[4])_$(radius_set[rad_ix])_auto"]
        full_params[1:2] = full_params[1:2] .|> round .|> Integer
        d = OrderedDict(zip(hpnames, full_params))
        p = d
        alias = "log x auto"
        mae_score, pmae_score, roc_score, proc_score = makesim(full_params..., 
                                                               datasets=datasets,
                                                               radius=radius_set[rad_ix],
                                                               #transform=x->log10.(x .+ 1e-30),
                                                               transform=x -> log10.(x .+ 1e-30),
                                                               transform_alias=alias,
                                                               #transform_back=x->x .^ 10 .- 1e-30)
                                                               transform_back=x->x .- 1e-30)
        d["transform"] = alias
        d["mae"] = round(mae_score,digits=10)
        d["pmae"] = pmae_score |> x->round(x, digits=8)
        #d["roc"] = roc_score |> x->round(x, digits=8)
        #d["proc"] = proc_score |> x->round(x, digits=8)
        d["radius"] = radius_set[rad_ix]
        @tagsave(datadir("sims","tuning", "time", savename(Dict(p), "jld2")), Dict(d))
        rad_ix -= 1
    end
end


df2 = collect_results(datadir("sims", "tuning", "time"))

sort!(df2, [:pmae, :mae])
filter!(x->x.savepath!="time_1", df2)
top_results = first(df2, 20)

data = readdlm(readdir(datadir("sims", "Adaptive"), join=true)[1], ',', header=false)
train_data = data[1:1, :] |> x-> log10.(x .+ 1e-30)

#= Compare the first N results to see which is truly best =#
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
        plot!(train_data[i, buffer:end], title="", label="Groud Truth", legend=:outertopright)
        savefig(projectdir("plots_time","species$(i)_$(j).png"))
    end
    j += 1
end

#= Plots best parameter set with varying start up information =#

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
    plot(output[1, :], title="", label="Predicted", legend=:outertopright)
    plot!(train_data[1, buffer:end], title="", label="Groud Truth", legend=:outertopright)
    savefig(projectdir("plots_time","time_$(buffer).png"))
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
        train_data = readdlm(ds_path, ',', header=false)[1:1,:] |> x -> log10.(x .+ 1e-30)
        weights_filename = ds_path |> basename |>
                x -> replace(x, "CVODE_" => "W_out_time_") |>
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
    serialize(path("W_time"), W)
    serialize(path("W_in_time"), W_in)
end