using DrWatson
@quickactivate "UCLCHEM Surrogate"

using Surrogates

include(srcdir("EchoStateNetwork.jl"))

using DifferentialEquations, Plots, Sundials

function rober(du,u,p,t)
  y₁,y₂,y₃ = u
  k₁,k₂,k₃ = p
  du[1] = -k₁*y₁+k₃*y₂*y₃
  du[2] =  k₁*y₁-k₂*y₂^2-k₃*y₂*y₃
  du[3] =  k₂*y₂^2
  nothing
end

rates = [0.04,3e7,1e4]
u0 = [1.0,1e-30,1e-30]
tspan = (0., 1e5)
prob = ODEProblem(rober, u0, tspan, rates)
sol = solve(prob, CVODE_BDF(), abstol=1e-16, reltol=1e-10)
high_error_sol = solve(prob, CVODE_BDF(), abstol=1e-6, reltol=1e-4)
train = hcat(sol.u...)
high_error_train = hcat(high_error_sol.u...)
rates_t = repeat(rates, length(sol.t)) |> x->reshape(x, length(rates), :)

X = train[:, 1:end-1]
high_error_X = high_error_train[:, 1:end-1]
y = train[:, 2:end]

warmup_length = 10
warmup = X[:, begin:warmup_length]
steps = size(y, 2) - size(warmup, 2)

"""
Simple Test of fit on the Robertson Problem
"""

# ESR
esn = ESN.EchoStateNetwork{Float64, ESN.EchoStateReservoir{Float64}}(3,300,3, ess=true);
desn = ESN.DeepEchoStateNetwork{Float64, ESN.EchoStateReservoir{Float64}}(3, 50, 6,3);

ESN.train!(esn, [X], [y], 1e-6)
ESN.train!(desn, [X], [y], 2e-7)

pred1 = ESN.predict!(esn, warmup, steps) |> x->vcat(x, hcat(sum.(eachcol(x))...))
pred2 = ESN.predict!(desn, warmup, steps) |> x->vcat(x, hcat(sum.(eachcol(x))...))
_y = y |> x->vcat(x, hcat(sum.(eachcol(x))...))

plot(sol.t[warmup_length+1:end], _y[:, warmup_length:end]', xscale=:log10, layout=4, legend=false)
plot!(sol.t[warmup_length+1:end], pred1', xscale=:log10, layout=4)
p1 = plot!(sol.t[warmup_length+1:end], pred2', xscale=:log10, layout=4)

savefig(p1, projectdir("images", "SS_prediction_RP_ESR.png"))

# SCR
esn = ESN.EchoStateNetwork{Float64, ESN.SimpleCycleReservoir{Float64}}(3,300,3; c=.7, ess=false);
desn = ESN.DeepEchoStateNetwork{Float64, ESN.SimpleCycleReservoir{Float64}}(3, 50, 6, 3, c=.65);

ESN.train!(esn, [X], [y], 2e-3)
ESN.train!(desn, [X], [y], 1e-5)

pred1 = ESN.predict!(esn, warmup, steps) |> x->vcat(x, hcat(sum.(eachcol(x))...))
pred2 = ESN.predict!(desn, warmup, steps) |> x->vcat(x, hcat(sum.(eachcol(x))...))
_y = y |> x->vcat(x, hcat(sum.(eachcol(x))...))

plot(sol.t[warmup_length+1:end], _y[:, warmup_length:end]', xscale=:log10, label="Ground Truth", layout=4)
plot!(sol.t[warmup_length+1:end], pred1', xscale=:log10, label="CTESN", layout=4)
p2 = plot!(sol.t[warmup_length+1:end], pred2', label="DeepCTESN", xscale=:log10, layout=4,legend=false)

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
MAE across thirty samples
"""
rates_set1 = sample(33, [0.04, 0.01,0.01], [1.0, 4e5, 9e4],SobolSample())
rates_set2 = sample(33, [0.04, 0.01,0.01], [4e5, 1.0, 9e4],SobolSample())
rates_set3 = sample(34, [0.04, 0.01,0.01], [4e5, 4e5, 1.0],SobolSample())
rates_set = [rates_set1; rates_set2; rates_set3]

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
X_set = full .|> x->x[begin]
y_set = full .|> x->x[end]

X = X_set[begin:begin]
y = y_set[begin:begin]

_y = y[begin] |> x->vcat(x, hcat(sum.(eachcol(x))...))

warmup_length = 10
warmup = X[begin][:, begin:warmup_length]
steps = size(_y, 2) - size(warmup, 2)

using Flux

function test!(esn, beta)
  ESN.train!(esn, X, y, beta)
  pred = ESN.predict!(esn, warmup, steps) |> x->vcat(x, hcat(sum.(eachcol(2 .^ x[1:end, :]))...))
  Flux.Losses.mae(pred, _y[begin:end, warmup_length:end])
end

function test_all(esn, beta=2.0, reduction_factor=.6)
    error = Inf
    while true
      new_error = test!(esn, beta)
      if new_error > error || isinf(new_error)
        return (error, beta / reduction_factor)
      else
        error = new_error
      end
      beta *= reduction_factor
    end
end



# ESR
esn = ESN.EchoStateNetwork{Float64, ESN.EchoStateReservoir{Float64}}(3,300,3);
desn = ESN.DeepEchoStateNetwork{Float64, ESN.EchoStateReservoir{Float64}}(3, 50, 6,3);

error_esr_ctesn = Float64[]
error_esr_deep_ctesn = Float64[]

err, esn_beta = test_all(esn, 100)
err, desn_beta = test_all(desn, 5)


for (x_i,y_i) in zip(X_set,y_set)
  warmup_length_i = 10
  warmup_i = x_i[:, begin:warmup_length_i]
  steps_i = size(y_i, 2) - size(warmup_i, 2)

  ESN.train!(esn, [x_i], [y_i], esn_beta)
  ESN.train!(desn, [x_i], [y_i], desn_beta)
  pred1 = ESN.predict!(esn, warmup_i, steps_i)
  pred2 = ESN.predict!(desn, warmup_i, steps_i) 
  push!(error_esr_ctesn, Flux.Losses.mae(pred1, y_i[:, warmup_length_i:end]))
  push!(error_esr_deep_ctesn, Flux.Losses.mae(pred2, y_i[:, warmup_length_i:end]))
end

# SCR
esn = ESN.EchoStateNetwork{Float64, ESN.SimpleCycleReservoir{Float64}}(3,300,3; c=.7);
desn = ESN.DeepEchoStateNetwork{Float64, ESN.SimpleCycleReservoir{Float64}}(3, 50, 6, 3, c=.65);

error_scr_ctesn = Float64[]
error_scr_deep_ctesn = Float64[]

err, esn_beta = test_all(esn, 100)
err, desn_beta = test_all(desn, 5)


for (x_i,y_i) in zip(X_set,y_set)
  warmup_length_i = 10
  warmup_i = x_i[:, begin:warmup_length_i]
  steps_i = size(y_i, 2) - size(warmup_i, 2)

  ESN.train!(esn, [x_i], [y_i], esn_beta)
  ESN.train!(desn, [x_i], [y_i], desn_beta)
  pred1 = ESN.predict!(esn, warmup_i, steps_i)
  pred2 = ESN.predict!(desn, warmup_i, steps_i) 
  push!(error_scr_ctesn, Flux.Losses.mae(pred1, y_i[:, warmup_length_i:end]))
  push!(error_scr_deep_ctesn, Flux.Losses.mae(pred2, y_i[:, warmup_length_i:end]))
end

# DLR
esn = ESN.EchoStateNetwork{Float64, ESN.DelayLineReservoir{Float64}}(3,300,3, c=.8);
desn = ESN.DeepEchoStateNetwork{Float64, ESN.DelayLineReservoir{Float64}}(3, 50, 6,3, c=.6);

error_dlr_ctesn = Float64[]
error_dlr_deep_ctesn = Float64[]


err, esn_beta = test_all(esn, 100)
err, desn_beta = test_all(desn, 5)

for (x_i,y_i) in zip(X_set,y_set)
  warmup_length_i = 10
  warmup_i = x_i[:, begin:warmup_length_i]
  steps_i = size(y_i, 2) - size(warmup_i, 2)

  ESN.train!(esn, [x_i], [y_i], esn_beta)
  ESN.train!(desn, [x_i], [y_i], desn_beta)
  pred1 = ESN.predict!(esn, warmup_i, steps_i)
  pred2 = ESN.predict!(desn, warmup_i, steps_i) 
  push!(error_dlr_ctesn, Flux.Losses.mae(pred1, y_i[:, warmup_length_i:end]))
  push!(error_dlr_deep_ctesn, Flux.Losses.mae(pred2, y_i[:, warmup_length_i:end]))
end

# DLRF
esn = ESN.EchoStateNetwork{Float64, ESN.DelayLineReservoir{Float64}}(3,300,3, c=.7, feedback=.3);
desn = ESN.DeepEchoStateNetwork{Float64, ESN.DelayLineReservoir{Float64}}(3, 50, 6,3, c=.7, feedback=.3);

error_dlrf_ctesn = Float64[]
error_dlrf_deep_ctesn = Float64[]

err, esn_beta = test_all(esn, 100)
err, desn_beta = test_all(desn, 5)


for (x_i,y_i) in zip(X_set,y_set)
  warmup_length_i = 10
  warmup_i = x_i[:, begin:warmup_length_i]
  steps_i = size(y_i, 2) - size(warmup_i, 2)

  ESN.train!(esn, [x_i], [y_i], esn_beta)
  ESN.train!(desn, [x_i], [y_i], desn_beta)
  pred1 = ESN.predict!(esn, warmup_i, steps_i)
  pred2 = ESN.predict!(desn, warmup_i, steps_i) 
  push!(error_dlrf_ctesn, Flux.Losses.mae(pred1, y_i[:, warmup_length_i:end]))
  push!(error_dlrf_deep_ctesn, Flux.Losses.mae(pred2, y_i[:, warmup_length_i:end]))
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
savefig(projectdir("images", "RB_SS_SR_log2_box_plot_validation_100_MAE.png"))


"""
ROC Error across 100 samples of the Robertson Problem
"""


function test!(esn, beta)
  ESN.train!(esn, X, y, beta)
  pred = ESN.predict!(esn, warmup, steps) |> x->vcat(x, hcat(sum.(eachcol(2 .^ x[1:end, :]))...))
  roc(pred, _y[begin:end, warmup_length:end])
end

function test_all(esn, beta=2.0, reduction_factor=.6)
    error = Inf
    while true
      new_error = test!(esn, beta)
      if new_error > error || isinf(new_error)
        return (error, beta / reduction_factor)
      else
        error = new_error
      end
      beta *= reduction_factor
    end
end



# ESR
esn = ESN.EchoStateNetwork{Float64, ESN.EchoStateReservoir{Float64}}(3,300,3);
desn = ESN.DeepEchoStateNetwork{Float64, ESN.EchoStateReservoir{Float64}}(3, 50, 6,3);

error_esr_ctesn = Float64[]
error_esr_deep_ctesn = Float64[]

err, esn_beta = test_all(esn, 100)
err, desn_beta = test_all(desn, 5)


for (x_i,y_i) in zip(X_set,y_set)
  warmup_length = 10
  warmup = x_i[:, begin:warmup_length]
  steps = size(y_i, 2) - size(warmup, 2)

  ESN.train!(esn, [x_i], [y_i], esn_beta)
  ESN.train!(desn, [x_i], [y_i], desn_beta)
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

err, esn_beta = test_all(esn, 100)
err, desn_beta = test_all(desn, 5)


for (x_i,y_i) in zip(X_set,y_set)
  warmup_length = 10
  warmup = x_i[:, begin:warmup_length]
  steps = size(y_i, 2) - size(warmup, 2)
  ESN.train!(esn, [x_i], [y_i], esn_beta)
  ESN.train!(desn, [x_i], [y_i], desn_beta)
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


err, esn_beta = test_all(esn, 100)
err, desn_beta = test_all(desn, 5)

for (x_i,y_i) in zip(X_set,y_set)
  warmup_length = 10
  warmup = x_i[:, begin:warmup_length]
  steps = size(y_i, 2) - size(warmup, 2)

  ESN.train!(esn, [x_i], [y_i], esn_beta)
  ESN.train!(desn, [x_i], [y_i], desn_beta)
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

err, esn_beta = test_all(esn, 100)
err, desn_beta = test_all(desn, 5)


for (x_i,y_i) in zip(X_set,y_set)
  warmup_length = 10
  warmup = x_i[:, begin:warmup_length]
  steps = size(y_i, 2) - size(warmup, 2)

  ESN.train!(esn, [x_i], [y_i], esn_beta)
  ESN.train!(desn, [x_i], [y_i], desn_beta)
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
savefig(projectdir("images", "RB_SS_SR_log2_box_plot_validation_100_ROC.png"))
