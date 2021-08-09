
"""
This is the Singleton Engine object.

It takes in a hollow model and then runs a series of Tests
using different sets of parameters

"""
struct Engine
    models::Vector{Model}
    tests::Vector{Test}
end

function run(test::Test, model)
    
end


function run(e::Engine, saveto=x->datadir("engine_results", x))
    isdir(saveto("")) || mkdir(saveto(""))

    alias = Dict()
    for model in e.models
        alias["model_type"] = typeof(model)
        while tune!(model)
            results = Dict()
            results["model_type"] = typeof(model)
            for test in e.tests
                results[typeof(test)] = run(test, model)
            end
            @tagsave(saveto("sims","tuning", "time", savename(alias, "jld2")), results)
        end
    end
end