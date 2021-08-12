using DrWatson
@quickactivate "UCLCHEM Surrogate"

using Surrogates

include(srcdir("EchoStateNetwork.jl"))

using DifferentialEquations, Plots

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
esn = ESN.EchoStateNetwork{Float64, ESN.EchoStateReservoir{Float64}}(3,300,3);
desn = ESN.DeepEchoStateNetwork{Float64, ESN.EchoStateReservoir{Float64}}(3, 50, 6,3);

ESN.train!(esn, [X], [y], 4e-5)
ESN.train!(desn, [X], [y], 2e-7)

pred1 = ESN.predict!(esn, warmup, steps) |> x->vcat(x, hcat(sum.(eachcol(x))...))
pred2 = ESN.predict!(desn, warmup, steps) |> x->vcat(x, hcat(sum.(eachcol(x))...))
_y = y |> x->vcat(x, hcat(sum.(eachcol(x))...))

plot(sol.t[warmup_length+1:end], _y[:, warmup_length:end]', xscale=:log10, label="GT", layout=4, legend=:outertopright)
plot!(sol.t[warmup_length+1:end], pred1', xscale=:log10, label="CTESN", layout=4)
plot!(sol.t[warmup_length+1:end], pred2', xscale=:log10, label="DeepCTESN", layout=4)

savefig(projectdir("images", "SS_prediction_RP_ESR.png"))

# SCR
esn = ESN.EchoStateNetwork{Float64, ESN.SimpleCycleReservoir{Float64}}(3,300,3; c=.7);
desn = ESN.DeepEchoStateNetwork{Float64, ESN.SimpleCycleReservoir{Float64}}(3, 50, 6, 3, c=.65);

ESN.train!(esn, [X], [y], 2e-11)
ESN.train!(desn, [X], [y], 1e-10)

pred1 = ESN.predict!(esn, warmup, steps) |> x->vcat(x, hcat(sum.(eachcol(x))...))
pred2 = ESN.predict!(desn, warmup, steps) |> x->vcat(x, hcat(sum.(eachcol(x))...))
_y = y |> x->vcat(x, hcat(sum.(eachcol(x))...))

plot(sol.t[warmup_length+1:end], _y[:, warmup_length:end]', xscale=:log10, label="GT", layout=4, legend=:outertopright)
plot!(sol.t[warmup_length+1:end], pred1', xscale=:log10, label="CTESN", layout=4)
plot!(sol.t[warmup_length+1:end], pred2', xscale=:log10, label="DeepCTESN", layout=4)

savefig(projectdir("images", "SS_prediction_RP_SCR.png"))

# DLR
esn = ESN.EchoStateNetwork{Float64, ESN.DelayLineReservoir{Float64}}(3,300,3, c=.8);
desn = ESN.DeepEchoStateNetwork{Float64, ESN.DelayLineReservoir{Float64}}(3, 50, 6,3, c=.6);

ESN.train!(esn, [X], [y], 5e-12)
ESN.train!(desn, [X], [y], 1e-11)

pred1 = ESN.predict!(esn, warmup, steps) |> x->vcat(x, hcat(sum.(eachcol(x))...))
pred2 = ESN.predict!(desn, warmup, steps) |> x->vcat(x, hcat(sum.(eachcol(x))...))
_y = y |> x->vcat(x, hcat(sum.(eachcol(x))...))

plot(sol.t[warmup_length+1:end], _y[:, warmup_length:end]', xscale=:log10, label="GT", layout=4, legend=:outertopright)
plot!(sol.t[warmup_length+1:end], pred1', xscale=:log10, label="CTESN", layout=4)
plot!(sol.t[warmup_length+1:end], pred2', xscale=:log10, label="DeepCTESN", layout=4)

savefig(projectdir("images", "SS_prediction_RP_DLR.png"))

# DLR w/F
esn = ESN.EchoStateNetwork{Float64, ESN.DelayLineReservoir{Float64}}(3,300,3, c=.7, feedback=.3);
desn = ESN.DeepEchoStateNetwork{Float64, ESN.DelayLineReservoir{Float64}}(3, 50, 6,3, c=.7, feedback=.3);

ESN.train!(esn, [X], [y], 1e-8)
ESN.train!(desn, [X], [y], 1e-8)

pred1 = ESN.predict!(esn, warmup, steps) |> x->vcat(x, hcat(sum.(eachcol(x))...))
pred2 = ESN.predict!(desn, warmup, steps) |> x->vcat(x, hcat(sum.(eachcol(x))...))
_y = y |> x->vcat(x, hcat(sum.(eachcol(x))...))

plot(sol.t[warmup_length+1:end], _y[:, warmup_length:end]', xscale=:log10, label="GT", layout=4, legend=:outertopright)
plot!(sol.t[warmup_length+1:end], pred1', xscale=:log10, label="CTESN", layout=4)
plot!(sol.t[warmup_length+1:end], pred2', xscale=:log10, label="DeepCTESN", layout=4)

savefig(projectdir("images", "SS_prediction_RP_DLRF.png"))