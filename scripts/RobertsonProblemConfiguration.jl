using DrWatson
@quickactivate "UCLCHEM Surrogate"

using Surrogates

include(srcdir("EchoStateNetwork.jl"))
include(srcdir("Scoring.jl"))

using DifferentialEquations, Plots, Sundials

function rober(du,u,p,t)
  y₁,y₂,y₃ = u
  k₁,k₂,k₃ = p
  du[1] = -k₁*y₁+k₃*y₂*y₃
  du[2] =  k₁*y₁-k₂*y₂^2-k₃*y₂*y₃
  du[3] =  k₂*y₂^2
  nothing
end

rates = [0.04,4e5,1e4]
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



# MAE across thirty samples

rates_set1 = sample(10, [0.04, 0.01,0.01], [1.0, 4e5, 9e4],SobolSample())
rates_set2 = sample(10, [0.04, 0.01,0.01], [4e5, 1.0, 9e4],SobolSample())
rates_set3 = sample(10, [0.04, 0.01,0.01], [4e5, 4e5, 1.0],SobolSample())
rates_set = [rates_set1; rates_set2; rates_set3]

@info "Solving sample set.."

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


@info "Starting ESR"
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

@info "Starting SCR"
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

@info "Starting DLR"
# DLR
esn = ESN.EchoStateNetwork{Float64, ESN.DelayLineReservoir{Float64}}(3,300,3, c=.7);
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

@info "Starting DLRF"
# DLRF
esn = ESN.EchoStateNetwork{Float64, ESN.DelayLineReservoir{Float64}}(3,300,3, c=.7, feedback=.1);
desn = ESN.DeepEchoStateNetwork{Float64, ESN.DelayLineReservoir{Float64}}(3, 50, 6,3, c=.7, feedback=.15);

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
savefig(projectdir("output", "MAE_Boxplots_RobertsonProblem.png"))


# ROC Error across 30 samples of the Robertson Problem



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


@info "Starting ROC Boxplots"

# ESR
@info "Starting ESR"
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
  push!(error_esr_ctesn, Flux.Losses.mae(pred1, y_i[:, warmup_length:end]))
  push!(error_esr_deep_ctesn, Flux.Losses.mae(pred2, y_i[:, warmup_length:end]))
end

@info "Starting SCR"
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
  push!(error_scr_ctesn, Flux.Losses.mae(pred1, y_i[:, warmup_length:end]))
  push!(error_scr_deep_ctesn, Flux.Losses.mae(pred2, y_i[:, warmup_length:end]))
end

@info "Starting DLR"
# DLR
esn = ESN.EchoStateNetwork{Float64, ESN.DelayLineReservoir{Float64}}(3,300,3, c=.7);
desn = ESN.DeepEchoStateNetwork{Float64, ESN.DelayLineReservoir{Float64}}(3, 50, 6,3, c=.65);

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
  push!(error_dlr_ctesn, Flux.Losses.mae(pred1, y_i[:, warmup_length:end]))
  push!(error_dlr_deep_ctesn, Flux.Losses.mae(pred2, y_i[:, warmup_length:end]))
end

@info "Starting DLRF"
# DLRF
esn = ESN.EchoStateNetwork{Float64, ESN.DelayLineReservoir{Float64}}(3,300,3, c=.7, feedback=.1);
desn = ESN.DeepEchoStateNetwork{Float64, ESN.DelayLineReservoir{Float64}}(3, 50, 6,3, c=.7, feedback=.15);

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
savefig(projectdir("output", "ROC_Boxplots_RobertsonProblem.png"))
