using DrWatson
@quickactivate "UCLCHEM Surrogate"

using Surrogates

include(srcdir("EchoStateNetwork.jl"))

using DifferentialEquations, Plots, Random

Random.seed!(0)

include.(srcdir.(["GasPhaseNetwork.jl", "CVODESolve.jl", "Visualize.jl"])) #, "NetworkSurrogate.jl"]))

rfp, icfp, sfp = map(x -> datadir("exp_raw", x), ["reactions_small.csv", "initcond0.csv", "species.csv"])
tspan = (0., 10^7 * 365. * 24. * 3600.)

rates_set_lower_bound = [1e-17, 0.5, 10, 1., 10., 1e2]
rates_set_upper_bound = [1e-14, 0.5, 300, 1., 10., 1e6]

parameter_samples = sample(1, rates_set_lower_bound, rates_set_upper_bound, SobolSample())[begin]
physical_parameters = Parameters(parameter_samples...)
cnp = formulate_all(rfp, icfp,physical_parameters, tspan=tspan)
prob = ODEProblem(cnp.network, cnp.u0, cnp.tspan)
@time sol = solve(prob, CVODE_BDF(), abstol=10e-30, reltol=10e-10);

train = hcat(sol.u...)
train .+= abs(minimum(train))
total_species = sum(train[:, begin])
train = eachcol(train) ./ total_species |> x->hcat(x...)

X = train[:, 1:end-1]
y = train[:, 2:end]

warmup_length = 10
warmup = X[:, begin:warmup_length]
steps = size(y, 2) - size(warmup, 2)
input_dimension = size(X, 1)
output_dimension = size(y, 1)

"""
Simple Test of fit on the Robertson Problem
"""

# ESR
esn = ESN.EchoStateNetwork{Float64, ESN.EchoStateReservoir{Float64}}(input_dimension,300,output_dimension);
desn = ESN.DeepEchoStateNetwork{Float64, ESN.EchoStateReservoir{Float64}}(input_dimension, 50, 6,output_dimension);

ESN.train!(esn, [X], [y], 4e-8)
ESN.train!(desn, [X], [y], 2e-7)

pred1 = ESN.predict!(esn, warmup, steps) |> x->vcat(x, hcat(sum.(eachcol(x))...))
pred2 = ESN.predict!(desn, warmup, steps) |> x->vcat(x, hcat(sum.(eachcol(x))...))
_y = y |> x->vcat(x, hcat(sum.(eachcol(x))...))

pred1 = (eachrow(pred1) .|> x-> x ./ x[begin]) |> x->hcat(x...)' |> Matrix
pred2 = (eachrow(pred2) .|> x-> x ./ x[begin]) |> x->hcat(x...)' |> Matrix
_y = (eachrow(_y) .|> x-> x ./ x[begin]) |> x->hcat(x...)' |> Matrix

plot(sol.t[warmup_length+1:end], _y[:, warmup_length:end]', xscale=:log10, layout=24, legend=false, size=(1200,1000))
plot!(sol.t[warmup_length+1:end], pred1', xscale=:log10, layout=24)
p1 = plot!(sol.t[warmup_length+1:end], pred2', xscale=:log10, layout=24)

savefig(p1, projectdir("images", "SS_prediction_RP_ESR.png"))

# SCR
esn = ESN.EchoStateNetwork{Float64, ESN.SimpleCycleReservoir{Float64}}(input_dimension,500,output_dimension; c=.7);
desn = ESN.DeepEchoStateNetwork{Float64, ESN.SimpleCycleReservoir{Float64}}(input_dimension, 75, 6, output_dimension, c=.65);

ESN.train!(esn, [X], [y], 5e-12)
ESN.train!(desn, [X], [y], 1e-7)

pred1 = ESN.predict!(esn, warmup, steps) |> x->vcat(x, hcat(sum.(eachcol(x))...))
pred2 = ESN.predict!(desn, warmup, steps) |> x->vcat(x, hcat(sum.(eachcol(x))...))
_y = y |> x->vcat(x, hcat(sum.(eachcol(x))...))

pred1 = (eachrow(pred1) .|> x-> x ./ x[begin]) |> x->hcat(x...)' |> Matrix
pred2 = (eachrow(pred2) .|> x-> x ./ x[begin]) |> x->hcat(x...)' |> Matrix
_y = (eachrow(_y) .|> x-> x ./ x[begin]) |> x->hcat(x...)' |> Matrix

plot(sol.t[warmup_length+1:end], _y[:, warmup_length:end]', xscale=:log10, layout=24, legend=false, size=(1200,1000))
plot!(sol.t[warmup_length+1:end], pred1', xscale=:log10, layout=24)
p2 = plot!(sol.t[warmup_length+1:end], pred2', xscale=:log10, layout=24)

savefig(p2, projectdir("images", "SS_prediction_RP_SCR.png"))

# DLR
esn = ESN.EchoStateNetwork{Float64, ESN.DelayLineReservoir{Float64}}(3,300,3, c=.8);
desn = ESN.DeepEchoStateNetwork{Float64, ESN.DelayLineReservoir{Float64}}(3, 50, 6,3, c=.6);

ESN.train!(esn, [X], [y], 5e-10)
ESN.train!(desn, [X], [y], 1e-11)

pred1 = ESN.predict!(esn, warmup, steps) |> x->vcat(x, hcat(sum.(eachcol(x))...))
pred2 = ESN.predict!(desn, warmup, steps) |> x->vcat(x, hcat(sum.(eachcol(x))...))
_y = y |> x->vcat(x, hcat(sum.(eachcol(x))...))

plot(sol.t[warmup_length+1:end], _y[:, warmup_length:end]', xscale=:log10, layout=4, legend=false)
plot!(sol.t[warmup_length+1:end], pred1', xscale=:log10, layout=4)
p3 = plot!(sol.t[warmup_length+1:end], pred2', xscale=:log10, layout=4)

savefig(p3, projectdir("images", "SS_prediction_RP_DLR.png"))

# DLR w/F
esn = ESN.EchoStateNetwork{Float64, ESN.DelayLineReservoir{Float64}}(3,300,3, c=.7, feedback=.3);
desn = ESN.DeepEchoStateNetwork{Float64, ESN.DelayLineReservoir{Float64}}(3, 50, 6,3, c=.7, feedback=.3);

ESN.train!(esn, [X], [y], 1e-8)
ESN.train!(desn, [X], [y], 1e-8)

pred1 = ESN.predict!(esn, warmup, steps) |> x->vcat(x, hcat(sum.(eachcol(x))...))
pred2 = ESN.predict!(desn, warmup, steps) |> x->vcat(x, hcat(sum.(eachcol(x))...))
_y = y |> x->vcat(x, hcat(sum.(eachcol(x))...))

plot(sol.t[warmup_length+1:end], _y[:, warmup_length:end]', xscale=:log10, layout=4, legend=false)
plot!(sol.t[warmup_length+1:end], pred1', xscale=:log10, layout=4)
p4 = plot!(sol.t[warmup_length+1:end], pred2', xscale=:log10, layout=4)

savefig(p4, projectdir("images", "SS_prediction_RP_DLRF.png"))

legend = plot([0 0 0 0], showaxis = false, grid = false, label = ["Ground Truth" "CTESN" "DeepCTESN"])
plot(p1, p2, p3, p4, legend, layout = @layout([[A B; C D] E{.1w}]))


"""
MSE across thirty samples
"""
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


rates_set1 = sample(33, [0.04, 0.01,0.01], [1.0, 4e5, 9e4],SobolSample())
rates_set2 = sample(33, [0.04, 0.01,0.01], [4e5, 1.0, 9e4],SobolSample())
rates_set3 = sample(34, [0.04, 0.01,0.01], [4e5, 4e5, 1.0],SobolSample())
rates_set = [rates_set1; rates_set2; rates_set3]
X = Matrix{Float64}[]
y = Matrix{Float64}[]

function solve_all(r)
    u = rand(3)
    u = u ./ sum(u)
    prob = ODEProblem(rober, u, tspan, [r...])
    sol = solve(prob, CVODE_BDF(), abstol=1e-20, reltol=1e-8)
    train = hcat(sol.u...)
    X_new = train[:, 1:end-1]
    y_new = train[:, 2:end]
    (X_new, y_new)
end

full = rates_set .|> solve_all 
X = full .|> x->x[begin]
y = full .|> x->x[end]


# ESR
esn = ESN.EchoStateNetwork{Float64, ESN.EchoStateReservoir{Float64}}(3,300,3);
desn = ESN.DeepEchoStateNetwork{Float64, ESN.EchoStateReservoir{Float64}}(3, 50, 6,3);

error_esr_ctesn = Float64[]
error_esr_deep_ctesn = Float64[]

for (x_i,y_i) in zip(X, y)
  warmup_length = 10
  warmup = x_i[:, begin:warmup_length]
  steps = size(y_i, 2) - size(warmup, 2)

  ESN.train!(esn, [x_i], [y_i], 4e-5)
  ESN.train!(desn, [x_i], [y_i], 2e-7)
  pred1 = ESN.predict!(esn, warmup, steps)
  pred2 = ESN.predict!(desn, warmup, steps) 
  push!(error_esr_ctesn, Flux.Losses.mae(pred1, y_i[:, warmup_length:end]))
  push!(error_esr_deep_ctesn, Flux.Losses.mae(pred2, y_i[:, warmup_length:end]))
end

# SCR
esn = ESN.EchoStateNetwork{Float64, ESN.SimpleCycleReservoir{Float64}}(3,300,3; c=.7);
desn = ESN.DeepEchoStateNetwork{Float64, ESN.SimpleCycleReservoir{Float64}}(3, 50, 6, 3, c=.65);

error_scr_ctesn = Float64[]
error_scr_deep_ctesn = Float64[]

for (x_i,y_i) in zip(X, y)
  warmup_length = 10
  warmup = x_i[:, begin:warmup_length]
  steps = size(y_i, 2) - size(warmup, 2)

  ESN.train!(esn, [x_i], [y_i], 2e-11)
  ESN.train!(desn, [x_i], [y_i], 1e-10)
  pred1 = ESN.predict!(esn, warmup, steps)
  pred2 = ESN.predict!(desn, warmup, steps) 
  push!(error_scr_ctesn, Flux.Losses.mae(pred1, y_i[:, warmup_length:end]))
  push!(error_scr_deep_ctesn, Flux.Losses.mae(pred2, y_i[:, warmup_length:end]))
end

# DLR
esn = ESN.EchoStateNetwork{Float64, ESN.DelayLineReservoir{Float64}}(3,300,3, c=.8);
desn = ESN.DeepEchoStateNetwork{Float64, ESN.DelayLineReservoir{Float64}}(3, 50, 6,3, c=.6);

error_dlr_ctesn = Float64[]
error_dlr_deep_ctesn = Float64[]

for (x_i,y_i) in zip(X, y)
  warmup_length = 10
  warmup = x_i[:, begin:warmup_length]
  steps = size(y_i, 2) - size(warmup, 2)

  ESN.train!(esn, [x_i], [y_i], 5e-10)
  ESN.train!(desn, [x_i], [y_i], 1e-11)
  pred1 = ESN.predict!(esn, warmup, steps)
  pred2 = ESN.predict!(desn, warmup, steps) 
  push!(error_dlr_ctesn, Flux.Losses.mae(pred1, y_i[:, warmup_length:end]))
  push!(error_dlr_deep_ctesn, Flux.Losses.mae(pred2, y_i[:, warmup_length:end]))
end

# DLRF
esn = ESN.EchoStateNetwork{Float64, ESN.DelayLineReservoir{Float64}}(3,300,3, c=.7, feedback=.3);
desn = ESN.DeepEchoStateNetwork{Float64, ESN.DelayLineReservoir{Float64}}(3, 50, 6,3, c=.7, feedback=.3);

error_dlrf_ctesn = Float64[]
error_dlrf_deep_ctesn = Float64[]

for (x_i,y_i) in zip(X, y)
  warmup_length = 10
  warmup = x_i[:, begin:warmup_length]
  steps = size(y_i, 2) - size(warmup, 2)

  ESN.train!(esn, [x_i], [y_i], 1e-8)
  ESN.train!(desn, [x_i], [y_i], 1e-8)
  pred1 = ESN.predict!(esn, warmup, steps)
  pred2 = ESN.predict!(desn, warmup, steps) 
  push!(error_dlrf_ctesn, Flux.Losses.mae(pred1, y_i[:, warmup_length:end]))
  push!(error_dlrf_deep_ctesn, Flux.Losses.mae(pred2, y_i[:, warmup_length:end]))
end


using StatsPlots

errors = [error_esr_ctesn, error_esr_deep_ctesn, error_scr_ctesn, error_scr_deep_ctesn, 
          error_dlr_ctesn, error_dlr_deep_ctesn, error_dlrf_ctesn, error_dlrf_deep_ctesn]

boxplot(errors[1], label="ESR CTESN", yscale=:log10, legend=:outertopright)
boxplot!(errors[2], label="ESR Deep CTESN")
boxplot!(errors[3], label="SCR CTESN")
boxplot!(errors[4], label="SCR Deep CTESN")
boxplot!(errors[5], label="DLR CTESN")
boxplot!(errors[6], label="DLR Deep CTESN")
boxplot!(errors[7], label="DLRF CTESN")
boxplot!(errors[8], label="DLRF Deep CTESN")
