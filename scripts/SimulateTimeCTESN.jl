using DrWatson
@quickactivate "UCLCHEM Surrogate"

include(srcdir("Simulation.jl"))
using Surrogates, DataStructures, JLD2, DataFrames

hyperparameter_lower_bound = [500, 10, 0.1, 0.0001, 0.001]
hyperparameter_upper_bound = [500, 400, .99, 10.0, 5.0]

hpnames = ["reservoir_size","degree","alpha", "sigma", "beta", "type", "savepath"]
hp_set = sample(1000, hyperparameter_lower_bound, hyperparameter_upper_bound, SobolSample())


for hp in hp_set
    @time begin
        full_params = [hp..., "time", "time_1"]
        full_params[1:2] = full_params[1:2] .|> round .|> Integer
        d = OrderedDict(zip(hpnames, full_params))
        p = d
        mae_score, pmae_score, roc_score, proc_score = makesim(full_params...)
        d["mae"] = mae_score
        d["pmae"] = pmae_score
        d["roc"] = roc_score
        d["proc"] = proc_score
        @tagsave(datadir("sims","tuning", d["type"], savename(Dict(p), "jld2")), Dict(d))
    end
end


df2 = collect_results(datadir("sims", "tuning", "time"))
sort!(df2, [:roc, :proc])

top_results = first(df2, 3)

data = readdlm(readdir(datadir("sims", "Adaptive"), join=true)[1], ',', header=false)
train_data = data[1:1, :] |> x-> log10.(x .+ 1e-30)
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
            extended_states = true)
    W_out = ESNtrain(esn, result.beta)
    esn = ESN(W,
            train_data[:,1:buffer],
            W_in,
            activation = tanh,
            alpha = result.alpha, 
            nla_type = NLADefault(), 
            extended_states = true)
    output = ESNpredict(esn, size(train_data, 2)-buffer, W_out)
    #println(sum.(eachcol(output)))
    for (i,k) in enumerate(eachrow(output))
        plot(output[i, :], title="", label="Predicted", legend=:outertopright)
        plot!(train_data[i, buffer:end], title="", label="Groud Truth", legend=:outertopright)
        savefig(projectdir("plots_time","species$(i)_$(j).png"))
    end
    j += 1
end
#=
function makesim(reservoir_size,
                 degree, 
                 alpha,
                 sigma,
                 beta,
                 type,
                 savepath,
                 datastore=adaptivedir(""),
                 transform=x->log10(x + 1),
                 transform_alias="log10",
                 transform_back=x->10^x-1,
                 sample_datapath=readdir(adaptivedir(""))[1],
                 radius=0.7,
                 activation = tanh, 
                 nla_type = NLADefault(), 
                 extended_states = false,
                 datatore=adaptivedir(""))
=#


