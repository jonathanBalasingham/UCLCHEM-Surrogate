using DrWatson
@quickactivate "UCLCHEM Surrogate"

using DifferentialEquations, Sundials, Serialization, Surrogates

using Random
Random.seed!(0)

include.(srcdir.(["GasPhaseNetwork.jl", "CVODESolve.jl", "Visualize.jl"])) #, "NetworkSurrogate.jl"]))

rfp, icfp, sfp = map(x -> datadir("exp_raw", x), ["reactions_final.csv", "initcond0.csv", "species.csv"])

include(srcdir("EchoStateNetwork.jl"))
include(srcdir("Scoring.jl"))

tspan = (0., 10^7 * 365. * 24. * 3600.)
                      #  zeta, omega, T, F_UV, A_v, E, density
rates_set_lower_bound = [1e-17, 0.5, 10, 1., 10., 1e2]
rates_set_upper_bound = [1e-14, 0.5, 300, 1., 10., 1e6]

dark_cloud_upper = get_rates(rfp, Parameters(rates_set_upper_bound...))
dark_cloud_lower = get_rates(rfp, Parameters(rates_set_lower_bound...))
true_dark_cloud_lower = [min(a,b) for (a,b) in zip(dark_cloud_lower, dark_cloud_upper)]
true_dark_cloud_upper = [max(a,b) for (a,b) in zip(dark_cloud_lower, dark_cloud_upper)]

bottom = filter(x->x>0,true_dark_cloud_lower) |> minimum |> log10 |> abs |> x->round(x)+1 
r(x) = replace(log10.(x) .+ bottom, -Inf=>0.0)

if isfile("./adaptive_simulation_dict")
    adaptive_simulation_dict = deserialize("./adaptive_simulation_dict")
else
    adaptive_simulation_dict = Dict()
end

serialize("./adaptive_simulation_dict", adaptive_simulation_dict)

adaptive_time_dict = values(adaptive_simulation_dict) .|> x->x[begin, :]
static_timepoints = adaptive_time_dict[begin]


if isfile("./static_simulation_dict")
    static_simulation_dict = deserialize("./static_simulation_dict")
else
    static_simulation_dict = Dict()
end

esn_adaptive_errors = Float64[]
desn_adaptive_errors = Float64[]
esn_adaptive_weight_dict = Dict()
desn_adaptive_weight_dict = Dict()

esn_beta = 1.9073486328125e-5
desn_beta = 9.5367431640625e-6

for (rates, solution) in adaptive_simulation_dict
    if rates in keys(esn_adaptive_weight_dict)
        continue
    end
    X = [solution[begin+1:end, begin:end-1]]
    y = [solution[begin+1:end, begin+1:end]]
    u0 = get_u0(X[begin])
    #esn_err, esn_beta = test_all(esn, X, y, nrm=true)
    #desn_err, desn_beta = test_all(desn, X, y, nrm=true)
    esn_err = test!(esn, esn_beta, X, y; nrm=true)
    desn_err = test!(desn, desn_beta, X, y; nrm=true)
    push!(esn_adaptive_errors, esn_err)
    push!(desn_adaptive_errors, desn_err)
    @info "Using esn beta: $esn_beta with mae error: $esn_err"
    @info "Using desn beta: $desn_beta with mae error: $desn_err"
    @info "MAX error for ESN: $(maximum(esn_adaptive_errors))"
    @info "MAX error for DESN: $(maximum(desn_adaptive_errors))"
    
    ESN.train!(esn, transform.(X, u0=u0), transform.(y, u0=u0), esn_beta)
    flattened_W_out = reshape(esn.output_layer.weight, :, 1)
    esn_adaptive_weight_dict[rates] = flattened_W_out
    @info "ESN Weight dictionary has $(length(keys(esn_adaptive_weight_dict))) entries"
    serialize("esn_adaptive_weight_dict", esn_adaptive_weight_dict)

    ESN.train!(desn, transform.(X, u0=u0), transform.(y, u0=u0), desn_beta)
    flattened_W_out = reshape(desn.output_layer.weight, :, 1)
    desn_adaptive_weight_dict[rates] = flattened_W_out
    @info "DESN Weight dictionary has $(length(keys(desn_adaptive_weight_dict))) entries"
    serialize("desn_adaptive_weight_dict", desn_adaptive_weight_dict)
end

esn_static_errors = Float64[]
desn_static_errors = Float64[]
esn_static_weight_dict = Dict()
desn_static_weight_dict = Dict()

esn_beta = 1.9073486328125e-5
desn_beta = 9.5367431640625e-6

for (rates, solution) in static_simulation_dict
    if rates in keys(esn_static_weight_dict)
        continue
    end
    X = [solution[begin+1:end, begin:end-1]]
    y = [solution[begin+1:end, begin+1:end]]
    u0 = get_u0(X[begin])
    esn_err = test!(esn, esn_beta, X, y; nrm=true)
    desn_err = test!(desn, desn_beta, X, y; nrm=true)
    push!(esn_static_errors, esn_err)
    push!(desn_static_errors, desn_err)
    @info "Using esn beta: $esn_beta with mae error: $esn_err"
    @info "Using desn beta: $desn_beta with mae error: $desn_err"
    @info "MAX error for ESN: $(maximum(esn_static_errors))"
    @info "MAX error for DESN: $(maximum(desn_static_errors))"
    

    push!(esn_static_errors, esn_err)
    push!(desn_static_errors, desn_err)
    @info "Using esn beta: $esn_beta with mae error: $esn_err"
    @info "Using desn beta: $desn_beta with mae error: $desn_err"

    ESN.train!(esn, transform.(X, u0=u0), transform.(y, u0=u0), esn_beta)
    flattened_W_out = reshape(esn.output_layer.weight, :, 1)
    esn_static_weight_dict[rates] = flattened_W_out
    @info "ESN Weight dictionary has $(length(keys(esn_static_weight_dict))) entries"
    serialize("esn_static_weight_dict", esn_static_weight_dict)

    ESN.train!(desn, transform.(X, u0=u0), transform.(y, u0=u0), desn_beta)
    flattened_W_out = reshape(desn.output_layer.weight, :, 1)
    desn_static_weight_dict[rates] = flattened_W_out
    @info "DESN Weight dictionary has $(length(keys(desn_static_weight_dict))) entries"
    serialize("desn_static_weight_dict", desn_static_weight_dict)
end

using Clustering

all_adaptive_data = hcat(values(adaptive_simulation_dict)...)
all_adaptive_data = all_adaptive_data[:, all_adaptive_data[begin, :] .!= 0] # filter out first point
all_adaptive_data[begin, :] = log2.(all_adaptive_data[begin, :]) # emphasizes time
clusters = kmeans(all_adaptive_data, 1500; maxiter=200, display=:iter)

new_timepoints = sort(2 .^ (clusters.centers[begin,:] ./ 10))



if isfile("./KNN1_simulation_dict")
    KNN1_simulation_dict = deserialize("./KNN1_simulation_dict")
else
    KNN1_simulation_dict = Dict()
end

esn_KNN1_errors = Float64[]
desn_KNN1_errors = Float64[]
esn_KNN1_weight_dict = Dict()
desn_KNN1_weight_dict = Dict()

esn_beta = 1.9073486328125e-5
desn_beta = 9.5367431640625e-6

for (rates, solution) in KNN1_simulation_dict
    if rates in keys(esn_KNN1_weight_dict)
        continue
    end
    X = [solution[begin+1:end, begin:end-1]]
    y = [solution[begin+1:end, begin+1:end]]
    u0 = get_u0(X[begin])
    esn_err = test!(esn, esn_beta, X, y; nrm=true)
    desn_err = test!(desn, desn_beta, X, y; nrm=true)
    push!(esn_KNN1_errors, esn_err)
    push!(desn_KNN1_errors, desn_err)
    @info "Using esn beta: $esn_beta with mae error: $esn_err"
    @info "Using desn beta: $desn_beta with mae error: $desn_err"
    @info "MAX error for ESN: $(maximum(esn_KNN1_errors))"
    @info "MAX error for DESN: $(maximum(desn_KNN1_errors))"
    

    push!(esn_KNN1_errors, esn_err)
    push!(desn_KNN1_errors, desn_err)
    @info "Using esn beta: $esn_beta with mae error: $esn_err"
    @info "Using desn beta: $desn_beta with mae error: $desn_err"

    ESN.train!(esn, transform.(X, u0=u0), transform.(y, u0=u0), esn_beta)
    flattened_W_out = reshape(esn.output_layer.weight, :, 1)
    esn_KNN1_weight_dict[rates] = flattened_W_out
    @info "ESN Weight dictionary has $(length(keys(esn_KNN1_weight_dict))) entries"
    serialize("esn_KNN1_weight_dict", esn_KNN1_weight_dict)

    ESN.train!(desn, transform.(X, u0=u0), transform.(y, u0=u0), desn_beta)
    flattened_W_out = reshape(desn.output_layer.weight, :, 1)
    desn_KNN1_weight_dict[rates] = flattened_W_out
    @info "DESN Weight dictionary has $(length(keys(desn_KNN1_weight_dict))) entries"
    serialize("desn_KNN1_weight_dict", desn_KNN1_weight_dict)
end

desn_beta = 1e-6

for (rates, solution) in KNN1_simulation_dict
    X = [solution[begin+1:end, begin:end-1]]
    y = [solution[begin+1:end, begin+1:end]]
    u0 = get_u0(X[begin])
    desn_err = test!(desn, desn_beta, X, y; nrm=true)
    push!(desn_KNN1_errors2, desn_err)
    @info "Using desn beta: $desn_beta with mae error: $desn_err"
    @info "MAX error for DESN: $(maximum(desn_KNN1_errors2))"
    ESN.train!(desn, transform.(X, u0=u0), transform.(y, u0=u0), desn_beta)
    flattened_W_out = reshape(desn.output_layer.weight, :, 1)
    desn_KNN1_weight_dict[rates] = flattened_W_out
    @info "DESN Weight dictionary has $(length(keys(desn_KNN1_weight_dict))) entries"
    #serialize("desn_KNN1_weight_dict", desn_KNN1_weight_dict)
end