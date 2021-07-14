include(srcdir("Simulation.jl"))
using Surrogates, DataStructures, JLD2, DataFrames

hyperparameter_lower_bound = [3000, 2000, 0.6, 0.5, 3.36006e-13]
hyperparameter_upper_bound = [3000, 2000, .99, .5, 0.1]

hpnames = ["reservoir_size","degree","alpha", "sigma", "beta", "type", "savepath"]
hp_set = sample(10, hyperparameter_lower_bound, hyperparameter_upper_bound, SobolSample())


for hp in hp_set
    @time begin
        full_params = [hp..., "species", "species_2_auto"]
        full_params[1:2] = full_params[1:2] .|> Integer
        d = OrderedDict(zip(hpnames, full_params))
        p = d
        alias = "log10: .+ 1e-30"
        mae_score, pmae_score, roc_score, proc_score = makesim(full_params..., 
                                                               transform=x->log10.(x .+ 1e-30),
                                                               transform_alias=alias,
                                                               transform_back=x->x .^ 10 .- 1e-30)
        d["transform"] = alias
        d["mae"] = mae_score
        d["pmae"] = pmae_score
        d["roc"] = roc_score
        d["proc"] = proc_score
        @tagsave(datadir("sims","tuning", "autonomous", savename(Dict(p), "jld2")), Dict(d))
    end
end

df = collect_results(datadir("sims", "tuning", "species"))
sort!(df, [:proc, :pmae])

W = readdlm(projectdir("_research", "W_species_1.csv"), ',', header=false)
W_in = readdlm(projectdir("_research", "W_in_species_1.csv"), ',', header=false)

top_results = first(df, 3)

data = readdlm(readdir(datadir("sims", "Adaptive"), join=true)[1], ',', header=false) # |> x-> log10.(x .+ 1.)
time = data[1, :]
train_data = data[2:end, :]
j = 1
for result in eachrow(top_results)
    esn = ESN(W,
            train_data,
            W_in,
            activation = tanh,
            alpha = result.alpha, 
            nla_type = NLADefault(), 
            extended_states = false)
    W_out = ESNtrain(esn, result.beta)
    output = ESNfitted(esn, W_out)
    for (i,k) in enumerate(eachrow(output))
        plot(time ./ (3600 * 24 * 365), output[i, :] .+ minimum(output[i,:]), title="", label="Predicted", legend=:outertopright)
        plot!(time ./ (3600 * 24 * 365), train_data[i, :] .+ minimum(train_data[i,:]), title="", label="Groud Truth", legend=:outertopright)
        ylims!(10^-30,1)
        xaxis!(:log10)
        xticks!([10^0, 10,10^2,10^3,10^4,10^5,10^6])
        yaxis!(:log10)
        xlims!(1,10^7)
        savefig(projectdir("plots","species$(i)_$(j).png"))
    end
    j += 1
end

sort!(df, [:pmae, :mae])

top_results = first(df, 3)
