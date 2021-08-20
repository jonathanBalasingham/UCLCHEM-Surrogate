using DifferentialEquations: collect
using DrWatson
@quickactivate "UCLCHEM Surrogate"

using DifferentialEquations, Sundials, Serialization, ReservoirComputing, Surrogates, Plots

using Random
Random.seed!(0)

include.(srcdir.(["GasPhaseNetwork.jl", "CVODESolve.jl", "Visualize.jl"])) #, "NetworkSurrogate.jl"]))

rfp, icfp, sfp = map(x -> datadir("exp_raw", x), ["reactions_small.csv", "initcond0.csv", "species.csv"])

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

parameter_samples = sample(3, true_dark_cloud_lower, true_dark_cloud_upper, SobolSample())

function condition(u, t, integrator)
    check_error(integrator) != :Success
end

function affect!(integrator)
    terminate!(integrator)
end


transform(x; u0) = hcat((eachrow(x) ./ u0)...)' |> Matrix
transform_back(x; u0) = hcat((eachrow(x) .* u0)...)' |> Matrix
get_u0(x) = x[:, begin]

callback = ContinuousCallback(condition, affect!)

timepoints = deserialize("./timepoints")
bottom = filter(x->x>0,true_dark_cloud_lower) |> minimum |> log10 |> abs |> x->round(x)+1 
r(x) = replace(log10.(x) .+ bottom, -Inf=>0.0)

# Get the ESN
esn = deserialize("esn")

# Select the surrogate & weight dictionary
weight_surrogate = deserialize("esn_weight_surrogate")
esn_weight_dict = deserialize("esn_weight_dict")

zetas = range(1e-14, stop=1e-17, length=5) |> collect
Ts = range(10, stop=300, length=10) |> collect
densities = range(1e2, stop=1e6, length=10) |> collect

function f(zeta, T, density; loss=Flux.Losses.mae)
    pa = Parameters(zeta, .5, T, 1., 10., density)
    p = formulate_all(rfp, icfp, pa, tspan=tspan)
    @time sol = solve(ODEProblem(p.network, p.u0, p.tspan), CVODE_BDF(), abstol=10e-30, reltol=10e-10, saveat=timepoints)
    train_subset = vcat(sol.t', hcat(sol.u...)) .|> x -> log2.(x .+ abs(minimum(x))*1.01)

    X = [train_subset[2:end, begin:end-1]]
    y = [train_subset[2:end, begin+1:end]]
    u0 = get_u0(X[begin])
    W_out_dims = size(esn.output_layer.weight)
    interp_rates = r(p.rates)
    W_out_interpolated = reshape(weight_surrogate(interp_rates), W_out_dims)
    esn.output_layer.weight .= W_out_interpolated

    warmup_length = 10
    warmup = X[begin][:, begin:warmup_length]
    steps = size(y[begin], 2) - size(warmup, 2)

    prediction = ESN.predict!(esn, transform(warmup, u0=u0), steps) |> x->transform_back(x, u0=u0)
    loss(y[begin][:, warmup_length:end], prediction)
end

hm = zeros(10,10)
for t in 1:length(Ts)
    for d in 1:length(densities)
        hm[t,d] = f(zetas[3], Ts[t], densities[d])
    end
end

p1 = heatmap(densities,Ts, hm, title="CTESN MAE", ylabel="T", xlabel="Density")

hm = zeros(10,10)
for t in 1:length(Ts)
    for d in 1:length(densities)
        hm[t,d] = f(zetas[3], Ts[t], densities[d], loss=roc)
    end
end

p2 = heatmap(densities,Ts, hm, title="CTESN ROC Error", ylabel="T", xlabel="Density")


hm = zeros(10,10)
for t in 1:length(Ts)
    for d in 1:length(densities)
        hm[t,d] = f(zetas[3], Ts[t], densities[d], loss=physical_error)
    end
end

p3 = heatmap(densities,Ts, hm, title="CTESN Physical Error", ylabel="T", xlabel="Density", size=(750, 500))

# Get the ESN
desn = deserialize("desn")

# Select the surrogate & weight dictionary
weight_surrogate = deserialize("desn_weight_surrogate")
desn_weight_dict = deserialize("desn_weight_dict")

function f(zeta, T, density; loss=Flux.Losses.mae)
    pa = Parameters(zeta, .5, T, 1., 10., density)
    p = formulate_all(rfp, icfp, pa, tspan=tspan)
    @time sol = solve(ODEProblem(p.network, p.u0, p.tspan), CVODE_BDF(), abstol=10e-30, reltol=10e-10, saveat=timepoints)
    train_subset = vcat(sol.t', hcat(sol.u...)) .|> x -> log2.(x .+ abs(minimum(x))*1.01)

    X = [train_subset[2:end, begin:end-1]]
    y = [train_subset[2:end, begin+1:end]]
    u0 = get_u0(X[begin])
    W_out_dims = size(desn.output_layer.weight)
    interp_rates = r(p.rates)
    W_out_interpolated = reshape(weight_surrogate(interp_rates), W_out_dims)
    desn.output_layer.weight .= W_out_interpolated

    warmup_length = 10
    warmup = X[begin][:, begin:warmup_length]
    steps = size(y[begin], 2) - size(warmup, 2)

    prediction = ESN.predict!(desn, transform(warmup, u0=u0), steps) |> x->transform_back(x, u0=u0)
    loss(y[begin][:, warmup_length:end], prediction)
end

hm = zeros(10,10)
for t in 1:length(Ts)
    for d in 1:length(densities)
        hm[t,d] = f(zetas[3], Ts[t], densities[d])
    end
end

p4 = heatmap(densities,Ts, hm, title="Deep CTESN MAE", ylabel="T", xlabel="Density")

hm = zeros(10,10)
for t in 1:length(Ts)
    for d in 1:length(densities)
        hm[t,d] = f(zetas[3], Ts[t], densities[d], loss=roc)
    end
end

p5 = heatmap(densities,Ts, hm, title="Deep CTESN ROC Error", ylabel="T", xlabel="Density")


hm = zeros(10,10)
for t in 1:length(Ts)
    for d in 1:length(densities)
        hm[t,d] = f(zetas[3], Ts[t], densities[d], loss=physical_error)
    end
end

p6 = heatmap(densities,Ts, hm, title="Deep CTESN Physical Error", ylabel="T", xlabel="Density", size=(750, 500))

l = @layout [p1 p4; p2 p5; p3 p6]
plot(p1,p4,p2,p5,p3,p6, layout=l)

savefig(projectdir("images", "SN_Interpolation_results_all.png"))


"""
Filtering out species less than log2(10-30)
"""


# Get the ESN
esn = deserialize("esn")

# Select the surrogate & weight dictionary
weight_surrogate = deserialize("esn_weight_surrogate")
esn_weight_dict = deserialize("esn_weight_dict")

zetas = range(1e-14, stop=1e-17, length=5) |> collect
Ts = range(10, stop=300, length=10) |> collect
densities = range(1e2, stop=1e6, length=10) |> collect

function f(zeta, T, density; loss=Flux.Losses.mae)
    pa = Parameters(zeta, .5, T, 1., 10., density)
    p = formulate_all(rfp, icfp, pa, tspan=tspan)
    @time sol = solve(ODEProblem(p.network, p.u0, p.tspan), CVODE_BDF(), abstol=10e-30, reltol=10e-10, saveat=timepoints)
    train_subset = vcat(sol.t', hcat(sol.u...)) .|> x -> log2.(x .+ abs(minimum(x))*1.01)

    X = [train_subset[2:end, begin:end-1]]
    y = [train_subset[2:end, begin+1:end]]
    u0 = get_u0(X[begin])
    W_out_dims = size(esn.output_layer.weight)
    interp_rates = r(p.rates)
    W_out_interpolated = reshape(weight_surrogate(interp_rates), W_out_dims)
    esn.output_layer.weight .= W_out_interpolated

    warmup_length = 10
    warmup = X[begin][:, begin:warmup_length]
    steps = size(y[begin], 2) - size(warmup, 2)
    inds = filter_to_significant_concentration(y[begin][:, warmup_length:end], :log2, indices_only=true)

    prediction = ESN.predict!(esn, transform(warmup, u0=u0), steps) |> x->transform_back(x, u0=u0)
    err = loss(y[begin][inds, warmup_length:end], prediction[inds, :])
    @show err
    err
end

hm = zeros(10,10)
for t in 1:length(Ts)
    for d in 1:length(densities)
        hm[t,d] = f(zetas[3], Ts[t], densities[d])
    end
end

p1 = heatmap(densities,Ts, hm, title="CTESN MAE", ylabel="T", xlabel="Density")

hm = zeros(10,10)
for t in 1:length(Ts)
    for d in 1:length(densities)
        hm[t,d] = f(zetas[3], Ts[t], densities[d], loss=roc)
    end
end

p2 = heatmap(densities,Ts, hm, title="CTESN ROC Error", ylabel="T", xlabel="Density")


hm = zeros(10,10)
for t in 1:length(Ts)
    for d in 1:length(densities)
        hm[t,d] = f(zetas[3], Ts[t], densities[d], loss=physical_error)
    end
end

p3 = heatmap(densities,Ts, hm, title="CTESN Physical Error", ylabel="T", xlabel="Density", size=(750, 500))

# Get the ESN
desn = deserialize("desn")

# Select the surrogate & weight dictionary
weight_surrogate = deserialize("desn_weight_surrogate")
desn_weight_dict = deserialize("desn_weight_dict")

function f(zeta, T, density; loss=Flux.Losses.mae)
    pa = Parameters(zeta, .5, T, 1., 10., density)
    p = formulate_all(rfp, icfp, pa, tspan=tspan)
    @time sol = solve(ODEProblem(p.network, p.u0, p.tspan), CVODE_BDF(), abstol=10e-30, reltol=10e-10, saveat=timepoints)
    train_subset = vcat(sol.t', hcat(sol.u...)) .|> x -> log2.(x .+ abs(minimum(x))*1.01)

    X = [train_subset[2:end, begin:end-1]]
    y = [train_subset[2:end, begin+1:end]]
    u0 = get_u0(X[begin])
    W_out_dims = size(desn.output_layer.weight)
    interp_rates = r(p.rates)
    W_out_interpolated = reshape(weight_surrogate(interp_rates), W_out_dims)
    desn.output_layer.weight .= W_out_interpolated

    warmup_length = 10
    warmup = X[begin][:, begin:warmup_length]
    steps = size(y[begin], 2) - size(warmup, 2)

    prediction = ESN.predict!(desn, transform(warmup, u0=u0), steps) |> x->transform_back(x, u0=u0)
    inds = filter_to_significant_concentration(y[begin][:, warmup_length:end], :log2, indices_only=true)
    loss(y[begin][inds, warmup_length:end], prediction[inds, :])
end

hm = zeros(10,10)
for t in 1:length(Ts)
    for d in 1:length(densities)
        hm[t,d] = f(zetas[3], Ts[t], densities[d])
    end
end

p4 = heatmap(densities,Ts, hm, title="Deep CTESN MAE", ylabel="T", xlabel="Density")

hm = zeros(10,10)
for t in 1:length(Ts)
    for d in 1:length(densities)
        hm[t,d] = f(zetas[3], Ts[t], densities[d], loss=roc)
    end
end

p5 = heatmap(densities,Ts, hm, title="Deep CTESN ROC Error", ylabel="T", xlabel="Density")


hm = zeros(10,10)
for t in 1:length(Ts)
    for d in 1:length(densities)
        hm[t,d] = f(zetas[3], Ts[t], densities[d], loss=physical_error)
    end
end

p6 = heatmap(densities,Ts, hm, title="Deep CTESN Physical Error", ylabel="T", xlabel="Density", size=(750, 500))

l = @layout [p1 p4; p2 p5; p3 p6]
plot(p1,p4,p2,p5,p3,p6, layout=l)

savefig(projectdir("images", "SN_Interpolation_results_filtered.png"))