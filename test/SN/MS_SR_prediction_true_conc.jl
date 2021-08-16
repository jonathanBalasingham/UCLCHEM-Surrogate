using DrWatson
@quickactivate "UCLCHEM Surrogate"

using Surrogates

include(srcdir("EchoStateNetwork.jl"))

using DifferentialEquations, Plots, Sundials

using Random
Random.seed!(0)

include.(srcdir.(["GasPhaseNetwork.jl", "CVODESolve.jl", "Visualize.jl"])) #, "NetworkSurrogate.jl"]))

rfp, icfp, sfp = map(x -> datadir("exp_raw", x), ["reactions_small.csv", "initcond0.csv", "species.csv"])


tspan = (0., 10^6 * 365. * 24. * 3600.)
                      #  zeta, omega, T, F_UV, A_v, E, density
rates_set_lower_bound = [1e-17, 0.5, 10, 1., 10., 1e2]
rates_set_upper_bound = [9.9e-17, 0.5, 300, 1., 10., 1e6]

parameter_samples1 = sample(2, rates_set_lower_bound, rates_set_upper_bound, SobolSample())

rates_set_lower_bound = [1e-16, 0.5, 10, 1., 10., 1e2]
rates_set_upper_bound = [9.9e-16, 0.5, 300, 1., 10., 1e6]

parameter_samples2 = sample(10, rates_set_lower_bound, rates_set_upper_bound, SobolSample())

rates_set_lower_bound = [1e-15, 0.5, 10, 1., 10., 1e2]
rates_set_upper_bound = [9.9e-15, 0.5, 300, 1., 10., 1e6]

parameter_samples3 = sample(10, rates_set_lower_bound, rates_set_upper_bound, SobolSample())

rates_set_lower_bound = [1e-14, 0.5, 10, 1., 10., 1e2]
rates_set_upper_bound = [9.9e-14, 0.5, 300, 1., 10., 1e6]

parameter_samples4 = sample(10, rates_set_lower_bound, rates_set_upper_bound, SobolSample())

parameter_samples = [parameter_samples1; parameter_samples2; parameter_samples3; parameter_samples4]

problems = parameter_samples4 .|>
            begin
                x->Parameters(x...) |> 
                x->formulate_all(rfp, icfp, x, tspan=tspan)
            end

for p in problems
    u0 = rand(length(p.u0))
    p.u0[begin:end] .= u0 / sum(u0)
end

train = problems .|> x->solve(x, abstol=10e-20);

full = parameter_samples .|>
            begin
                x->Parameters(x...) |> 
                x->formulate_all(rfp, icfp, x, tspan=tspan) |>
                x->
                begin 
                  prob=ODEProblem(x.network, x.u0, x.tspan)
                  @time sol = solve(prob, CVODE_BDF(), abstol=10e-20, reltol=10e-8)
                  (remake(prob, p=x.rates), sol)
                end
            end

solutions = problems .|> x->@time  solve(x, CVODE_BDF(), abstol=10e-30, reltol=10e-8)

# two possible pre-processes: each species between 0,1 
rates_length = length(train[begin].rates)

#preprocess1(train_set::Matrix) = train_set[rates_length+2:end, :] |> x -> (x .- minimum(x)) ./ (maximum(x) - minimum(x))
#preprocess2(train_set::Matrix) = eachcol(train_set) .|> x->x ./ sum(abs.(x))

full_train = full .|>
    sol->vcat(repeat([log10.(sol[begin].p)...], length(sol[end].t)) |> 
    x->reshape(x, length(sol[begin].p), :), log10.(sol[end].t)', hcat(sol[end].u...)) 

X = full_train .|> x->replace(x[:, begin:end-1], -Inf=>0.0)
y = full_train .|> x->replace(x[rates_length+2:end, 2:end], -Inf=>0.0)

input_dimension = size(X[begin], 1)
output_dimension = size(y[begin], 1)

test_ind = length(y)

warmup_length = 10
warmup = X[test_ind][:, begin:warmup_length]
steps = size(y[test_ind], 2) - size(warmup, 2)

"""
Prediction on Multiple Systems of a Small Network
"""

# ESR
esn = ESN.EchoStateNetwork{Float64, ESN.EchoStateReservoir{Float64}}(input_dimension, 500, output_dimension);
#desn = ESN.DeepEchoStateNetwork{Float64, ESN.EchoStateReservoir{Float64}}(7, 50, 6,4);
#sesn = ESN.SplitEchoStateNetwork{Float64, ESN. EchoStateReservoir{Float64}}((3,4), (200, 300), 4)
st = X[test_ind][begin:3, :]
xt = warmup[4:end, :]

ESN.train!(esn, X[1:end-1], y[1:end-1], 1e-3)
pred1 = ESN.predict!(esn, xt, st) |> x->vcat(x, hcat(sum.(eachcol(x[2:end, :]))...))

#ESN.train!(sesn, X[1:end-1], y[1:end-1], 3e-2)
#pred2 = ESN.predict!(sesn, xt, st) |> x->vcat(x, hcat(sum.(eachcol(x[2:end, :]))...))

_y = y[test_ind] |> x->vcat(x, hcat(sum.(eachcol(x[2:end, :]))...))

plot(10 .^ _y[begin, warmup_length:end], _y[2:end, warmup_length:end]', xscale=:log10, label="GT", layout=4, legend=:outertopright)
plot!(10 .^ pred1[begin, :], pred1[2:end, :]', xscale=:log10, label="CTESN", layout=4)
#plot!(10 .^ pred2[begin, :], pred2[2:end, :]', xscale=:log10, label="Split CTESN", layout=4)

savefig(projectdir("images", "MS_SR_TC_prediction_RP_ESR.png"))

# SCR
esn = ESN.EchoStateNetwork{Float64, ESN.SimpleCycleReservoir{Float64}}(input_dimension, 800, output_dimension; c=.7);
desn = ESN.DeepEchoStateNetwork{Float64, ESN.SimpleCycleReservoir{Float64}}(input_dimension, 200, 8, output_dimension,1.0, c=.7);

st = X[test_ind][begin:rates_length+1, :]
xt = warmup[rates_length+2:end, :]

ESN.train!(esn, X[1:end-1], y[1:end-1], 5.)
pred1 = ESN.predict!(esn, xt, st) |> x->vcat(x, hcat(sum.(eachcol(x[2:end, :]))...))

ESN.train!(desn, X[1:end-1], y[1:end-1], 1.)
pred2 = ESN.predict!(desn, xt, st) |> x->vcat(x, hcat(sum.(eachcol(x[2:end, :]))...))

_y = y[test_ind] |> x->vcat(x, hcat(sum.(eachcol(x[2:end, :]))...))

plot(10 .^ X[test_ind][1, warmup_length:end], _y[1:end, warmup_length:end]',
    xscale=:log10, label="GT", layout=length(train[begin].species)+1, legend=:outertopright, size=(1200,1000))
plot!(10 .^ X[test_ind][1, warmup_length+1:end], pred1[1:end, :]', xscale=:log10, label="CTESN", layout=length(train[begin].species)+1)
plot!(10 .^ X[test_ind][1, warmup_length+1:end], pred2[1:end, :]', xscale=:log10, label="DeepCTESN", layout=length(train[begin].species)+1)

savefig(projectdir("images", "MS_SR_TC_prediction_RP_SCR.png"))

# DLR

esn = ESN.EchoStateNetwork{Float64, ESN.DelayLineReservoir{Float64}}(input_dimension, 2000, output_dimension; c=.7);
desn = ESN.DeepEchoStateNetwork{Float64, ESN.DelayLineReservoir{Float64}}(input_dimension, 200, 5, output_dimension,1.0, c=.7);
#sesn = ESN.SplitEchoStateNetwork{Float64, ESN.DelayLineReservoir{Float64}}((rates_length+1, length(train[begin].species)), (1200, 800), output_dimension, c=.7)

st = X[test_ind][begin:rates_length+1, :]
xt = warmup[rates_length+2:end, :]

ESN.train!(esn, X[1:end-1], y[1:end-1], 1.0)
pred1 = ESN.predict!(esn, xt, st) |> x->vcat(x, hcat(sum.(eachcol(x[2:end, :]))...))

ESN.train!(desn, X[1:end-1], y[1:end-1], 2.0)
pred2 = ESN.predict!(desn, xt, st) |> x->vcat(x, hcat(sum.(eachcol(x[2:end, :]))...))

_y = y[test_ind] |> x->vcat(x, hcat(sum.(eachcol(x[2:end, :]))...))

plot(10 .^ X[test_ind][rates_length+1, warmup_length:end], _y[1:end, warmup_length:end]',
    xscale=:log10, label="GT", layout=length(train[begin].species)+1, legend=:outertopright, size=(1200,1000))
plot!(10 .^ X[test_ind][rates_length+1, warmup_length+1:end], pred1[1:end, :]', xscale=:log10, label="CTESN", layout=length(train[begin].species)+1)
plot!(10 .^ X[test_ind][rates_length+1, warmup_length+1:end], pred2[1:end, :]', xscale=:log10, label="DeepCTESN", layout=length(train[begin].species)+1)
plot!(10 .^ X[test_ind][rates_length+1, warmup_length+1:end], pred3[1:end, :]', xscale=:log10, label="SESN", layout=length(train[begin].species)+1)

savefig(projectdir("images", "MS_SR_TC_prediction_RP_DLR.png"))

# DLR w/F
esn = ESN.EchoStateNetwork{Float64, ESN.DelayLineReservoir{Float64}}(7,600,4, c=.7, feedback=.2);
desn = ESN.DeepEchoStateNetwork{Float64, ESN.DelayLineReservoir{Float64}}(7, 100, 6,4, c=.7, feedback=.1);

st = X[test_ind][begin:3, :]
xt = warmup[4:end, :]

ESN.train!(esn, X[1:end-1], y[1:end-1], 1e-3)
pred1 = ESN.predict!(esn, xt, st) |> x->vcat(x, hcat(sum.(eachcol(x[2:end, :]))...))

ESN.train!(desn, X[1:end-1], y[1:end-1], 8e-7)
pred2 = ESN.predict!(desn, xt, st) |> x->vcat(x, hcat(sum.(eachcol(x[2:end, :]))...))

_y = y[test_ind] |> x->vcat(x, hcat(sum.(eachcol(x[2:end, :]))...))

plot(10 .^ _y[begin, warmup_length:end], _y[2:end, warmup_length:end]', xscale=:log10, label="GT", layout=4, legend=:outertopright)
plot!(10 .^ pred1[begin, :], pred1[2:end, :]', xscale=:log10, label="CTESN", layout=4)
plot!(10 .^ pred2[begin, :], pred2[2:end, :]', xscale=:log10, label="DeepCTESN", layout=4)

savefig(projectdir("images", "MS_SR_TC_prediction_RP_DLRF.png"))

"""
MAE accross the validation set
"""
rates_set1 = sample(50, [0.04, 0.01,0.01], [1.0, 4e5, 9e4],SobolSample())
rates_set2 = sample(50, [0.04, 0.01,0.01], [4e5, 1.0, 9e4],SobolSample())
rates_set3 = sample(50, [0.04, 0.01,0.01], [4e5, 4e5, 1.0],SobolSample())
rates_set = [rates_set1; rates_set2; rates_set3]

function solve_all(r)
    u = rand(3)
    u = u ./ sum(u)
    prob = ODEProblem(rober, u, tspan, [r...])
    sol = solve(prob, CVODE_BDF(), abstol=1e-20, reltol=1e-8)
    train = hcat(sol.u...)
    rates_t = repeat([r...], length(sol.t)) |> x->reshape(x, length(rates), :)

    species_and_rates = vcat(rates_t, sol.t') |> x->log10.(x .+ 1e-30) |> x->vcat(x, train)
    X_new = species_and_rates[:, 1:end-1]
    y_new = species_and_rates[4:7, 2:end]
    (X_new, y_new)
end

full = rates_set .|> solve_all 
X_val = full .|> x->x[begin]
y_val = full .|> x->x[end]


# ESR
esn = ESN.EchoStateNetwork{Float64, ESN.EchoStateReservoir{Float64}}(7,500,4);
ESN.train!(esn, X, y, 1e-3)

error_esr_ctesn = Float64[]

for (x_i,y_i) in zip(X_val, y_val)
  warmup_length = 10
  warmup = x_i[:, begin:warmup_length]
  st = x_i[begin:3, :]
  xt = warmup[4:end, :]

  pred1 = ESN.predict!(esn, xt, st)
  push!(error_esr_ctesn, Flux.Losses.mae(pred1, y_i[:, warmup_length+1:end]))
end

# SCR
esn = ESN.EchoStateNetwork{Float64, ESN.SimpleCycleReservoir{Float64}}(7,600,4; c=.6);
desn = ESN.DeepEchoStateNetwork{Float64, ESN.SimpleCycleReservoir{Float64}}(7, 100, 6, 4, c=.7);

ESN.train!(esn, X, y, 1e-6)
ESN.train!(desn, X, y, 2e-11)

error_scr_ctesn = Float64[]
error_scr_deep_ctesn = Float64[]

for (x_i,y_i) in zip(X_val, y_val)
    warmup_length = 10
    warmup = x_i[:, begin:warmup_length]
    st = x_i[begin:3, :]
    xt = warmup[4:end, :]
  
    pred1 = ESN.predict!(esn, xt, st) 
    pred2 = ESN.predict!(desn, xt, st) 
    push!(error_scr_ctesn, Flux.Losses.mae(pred1, y_i[:, warmup_length+1:end]))
    push!(error_scr_deep_ctesn, Flux.Losses.mae(pred2, y_i[:, warmup_length+1:end]))
end

# DLR
esn = ESN.EchoStateNetwork{Float64, ESN.DelayLineReservoir{Float64}}(7,600,4, c=.6);
desn = ESN.DeepEchoStateNetwork{Float64, ESN.DelayLineReservoir{Float64}}(7, 100, 6,4, c=.6);

ESN.train!(esn, X, y, 4e-8)
ESN.train!(desn, X, y, 9e-11)

error_dlr_ctesn = Float64[]
error_dlr_deep_ctesn = Float64[]

for (x_i,y_i) in zip(X, y)
  warmup_length = 10
  warmup = x_i[:, begin:warmup_length]
  st = x_i[begin:3, :]
  xt = warmup[4:end, :]
  pred1 = ESN.predict!(esn, xt, st) 
  pred2 = ESN.predict!(desn, xt, st) 
  push!(error_dlr_ctesn, Flux.Losses.mae(pred1, y_i[:, warmup_length+1:end]))
  push!(error_dlr_deep_ctesn, Flux.Losses.mae(pred2, y_i[:, warmup_length+1:end]))
end

# DLRF
esn = ESN.EchoStateNetwork{Float64, ESN.DelayLineReservoir{Float64}}(7,600,4, c=.7, feedback=.2);
desn = ESN.DeepEchoStateNetwork{Float64, ESN.DelayLineReservoir{Float64}}(7, 100, 6,4, c=.7, feedback=.1);

ESN.train!(esn, X, y, 1e-3)
ESN.train!(desn, X, y, 8e-7)

error_dlrf_ctesn = Float64[]
error_dlrf_deep_ctesn = Float64[]

for (x_i,y_i) in zip(X, y)
  warmup_length = 10
  warmup = x_i[:, begin:warmup_length]
  st = x_i[begin:3, :]
  xt = warmup[4:end, :]
  pred1 = ESN.predict!(esn, xt, st) 
  pred2 = ESN.predict!(desn, xt, st) 
  push!(error_dlrf_ctesn, Flux.Losses.mae(pred1, y_i[:, warmup_length+1:end]))
  push!(error_dlrf_deep_ctesn, Flux.Losses.mae(pred2, y_i[:, warmup_length+1:end]))
end


using StatsPlots

errors = [error_esr_ctesn, error_scr_ctesn, error_scr_deep_ctesn, 
          error_dlr_ctesn, error_dlr_deep_ctesn, error_dlrf_ctesn, error_dlrf_deep_ctesn]

boxplot(errors[1], label="ESR CTESN", yscale=:log10, legend=:outertopright)
boxplot!(errors[2], label="SCR CTESN")
boxplot!(errors[3], label="SCR Deep CTESN")
boxplot!(errors[4], label="DLR CTESN")
boxplot!(errors[5], label="DLR Deep CTESN")
boxplot!(errors[6], label="DLRF CTESN")
boxplot!(errors[7], label="DLRF Deep CTESN")
