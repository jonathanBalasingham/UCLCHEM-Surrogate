include(srcdir("Simulation.jl"))
using Surrogates, DataStructures, JLD2, DataFrames

hyperparameter_lower_bound = [1000, 500, 0.1, 0.0001, 0.000000000001]
hyperparameter_upper_bound = [1000, 500, .99, 1.0, 0.1]

hpnames = ["reservoir_size","degree","alpha", "sigma", "beta", "type", "savepath"]
hp_set = sample(1000, hyperparameter_lower_bound, hyperparameter_upper_bound, SobolSample())


for hp in hp_set
    @time begin
        full_params = [hp..., "time", "time_1"]
        full_params[1:2] = full_params[1:2] .|> Integer
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


