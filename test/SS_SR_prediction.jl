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

species_and_rates = vcat(rates_t, sol.t', train) |> x->log10.(x .+ 1e-30)
X = species_and_rates[:, 1:end-1]
y = species_and_rates[4:7, 2:end]


warmup_length = 10
warmup = X[:, begin:warmup_length]
steps = size(y, 2) - size(warmup, 2)

"""
Simple Test of fit on the Robertson Problem
"""

# ESR
esn = ESN.EchoStateNetwork{Float64, ESN.EchoStateReservoir{Float64}}(7,300,4);
desn = ESN.DeepEchoStateNetwork{Float64, ESN.EchoStateReservoir{Float64}}(7, 50, 6,4);

ESN.train!(esn, [X], [y], 4e-5)
ESN.train!(desn, [X], [y], 2e-4)

st = X[begin:3, :]
xt = warmup[4:end, :]

pred1 = ESN.predict!(esn, xt, st) |> x->vcat(x, hcat(sum.(eachcol(10 .^ x[2:end, :]))...))
pred2 = ESN.predict!(desn, xt, st) |> x->vcat(x, hcat(sum.(eachcol(10 .^ x[2:end, :]))...))
_y = y |> x->vcat(x, hcat(sum.(eachcol(10 .^ x[2:end, :]))...))

plot(sol.t[warmup_length+1:end], _y[2:end, warmup_length:end]', xscale=:log10, label="GT", layout=4, legend=:outertopright)
plot!(10 .^ pred1[begin, :], pred1[2:end, :]', xscale=:log10, label="CTESN", layout=4)
plot!(10 .^ pred2[begin, :], pred2[2:end, :]', xscale=:log10, label="DeepCTESN", layout=4)

savefig(projectdir("images", "SS_SR_prediction_RP_ESR.png"))

# SCR
esn = ESN.EchoStateNetwork{Float64, ESN.SimpleCycleReservoir{Float64}}(7,300,4; c=.6);
desn = ESN.DeepEchoStateNetwork{Float64, ESN.SimpleCycleReservoir{Float64}}(7, 50, 6, 4, c=.6);

ESN.train!(esn, [X], [y], 6e-5)
ESN.train!(desn, [X], [y], 3e-3)

st = X[begin:3, :]
xt = warmup[4:end, :]

pred1 = ESN.predict!(esn, xt, st) |> x->vcat(x, hcat(sum.(eachcol(10 .^ x[2:end, :]))...))
pred2 = ESN.predict!(desn, xt, st) |> x->vcat(x, hcat(sum.(eachcol(10 .^ x[2:end, :]))...))
_y = y |> x->vcat(x, hcat(sum.(eachcol(10 .^ x[2:end, :]))...))

plot(sol.t[warmup_length+1:end], _y[2:end, warmup_length:end]', xscale=:log10, label="GT", layout=4, legend=:outertopright)
plot!(10 .^ pred1[begin, :], pred1[2:end, :]', xscale=:log10, label="CTESN", layout=4)
plot!(10 .^ pred2[begin, :], pred2[2:end, :]', xscale=:log10, label="DeepCTESN", layout=4)

savefig(projectdir("images", "SS_SR_prediction_RP_SCR.png"))

# DLR
esn = ESN.EchoStateNetwork{Float64, ESN.DelayLineReservoir{Float64}}(7,300,4, c=.6);
desn = ESN.DeepEchoStateNetwork{Float64, ESN.DelayLineReservoir{Float64}}(7, 50, 6,4, c=.6);

ESN.train!(esn, [X], [y], 5e-4)
ESN.train!(desn, [X], [y], 2e-3)

st = X[begin:3, :]
xt = warmup[4:end, :]

pred1 = ESN.predict!(esn, xt, st) |> x->vcat(x, hcat(sum.(eachcol(10 .^ x[2:end, :]))...))
pred2 = ESN.predict!(desn, xt, st) |> x->vcat(x, hcat(sum.(eachcol(10 .^ x[2:end, :]))...))
_y = y |> x->vcat(x, hcat(sum.(eachcol(10 .^ x[2:end, :]))...))

plot(sol.t[warmup_length+1:end], _y[2:end, warmup_length:end]', xscale=:log10, label="GT", layout=4, legend=:outertopright)
plot!(10 .^ pred1[begin, :], pred1[2:end, :]', xscale=:log10, label="CTESN", layout=4)
plot!(10 .^ pred2[begin, :], pred2[2:end, :]', xscale=:log10, label="DeepCTESN", layout=4)

savefig(projectdir("images", "SS_SR_prediction_RP_DLR.png"))

# DLR w/F
esn = ESN.EchoStateNetwork{Float64, ESN.DelayLineReservoir{Float64}}(7,300,4, c=.6, feedback=.2);
desn = ESN.DeepEchoStateNetwork{Float64, ESN.DelayLineReservoir{Float64}}(7, 50, 6,4, c=.6, feedback=.3);

ESN.train!(esn, [X], [y], 8e-4)
ESN.train!(desn, [X], [y], 2e-3)

st = X[begin:3, :]
xt = warmup[4:end, :]

pred1 = ESN.predict!(esn, xt, st) |> x->vcat(x, hcat(sum.(eachcol(10 .^ x[2:end, :]))...))
pred2 = ESN.predict!(desn, xt, st) |> x->vcat(x, hcat(sum.(eachcol(10 .^ x[2:end, :]))...))
_y = y |> x->vcat(x, hcat(sum.(eachcol(10 .^ x[2:end, :]))...))

plot(sol.t[warmup_length+1:end], _y[2:end, warmup_length:end]', xscale=:log10, label="GT", layout=4, legend=:outertopright)
plot!(10 .^ pred1[begin, :], pred1[2:end, :]', xscale=:log10, label="CTESN", layout=4)
plot!(10 .^ pred2[begin, :], pred2[2:end, :]', xscale=:log10, label="DeepCTESN", layout=4)

savefig(projectdir("images", "SS_SR_prediction_RP_DLRF.png"))
