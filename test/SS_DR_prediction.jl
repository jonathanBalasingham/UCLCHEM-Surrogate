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
"""
Dynamic Rates
"""
using DiffEqCallbacks
condition(u,t,integrator) = true
params = Vector[]
ending_rates = [10., 3e6, 1e2]

r(t) = rates .+ (t/ tspan[2])*(ending_rates .- rates)
function affect!(integrator)
    integrator.p .= r(integrator.t)
    push!(params, integrator.p)
end
cb = DiscreteCallback(condition,affect!)

prob = ODEProblem(rober, u0, tspan, rates, callback=cb)
sol = solve(prob, CVODE_BDF(), abstol=1e-16, reltol=1e-10)
train = hcat(sol.u...)

X = train[:, 1:end-1]
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

plot(sol.t[warmup_length+1:end], _y[:, warmup_length:end]', xscale=:log10, layout=4, legend=false)
plot!(sol.t[warmup_length+1:end], pred1', xscale=:log10, layout=4)
p1 = plot!(sol.t[warmup_length+1:end], pred2', xscale=:log10, layout=4)

savefig(p1, projectdir("images", "SS_MRprediction_RP_ESR.png"))

# SCR
esn = ESN.EchoStateNetwork{Float64, ESN.SimpleCycleReservoir{Float64}}(3,300,3; c=.7);
desn = ESN.DeepEchoStateNetwork{Float64, ESN.SimpleCycleReservoir{Float64}}(3, 50, 6, 3, c=.65);

ESN.train!(esn, [X], [y], 2e-11)
ESN.train!(desn, [X], [y], 1e-10)

pred1 = ESN.predict!(esn, warmup, steps) |> x->vcat(x, hcat(sum.(eachcol(x))...))
pred2 = ESN.predict!(desn, warmup, steps) |> x->vcat(x, hcat(sum.(eachcol(x))...))
_y = y |> x->vcat(x, hcat(sum.(eachcol(x))...))

plot(sol.t[warmup_length+1:end], _y[:, warmup_length:end]', xscale=:log10, label="Ground Truth", layout=4)
plot!(sol.t[warmup_length+1:end], pred1', xscale=:log10, label="CTESN", layout=4)
p2 = plot!(sol.t[warmup_length+1:end], pred2', label="DeepCTESN", xscale=:log10, layout=4,legend=false)

savefig(p2, projectdir("images", "SS_MRprediction_RP_SCR.png"))

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

savefig(p3, projectdir("images", "SS_MRprediction_RP_DLR.png"))

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

savefig(p4, projectdir("images", "SS_MRprediction_RP_DLRF.png"))

legend = plot([0 0 0 0], showaxis = false, grid = false, label = ["Ground Truth" "CTESN" "DeepCTESN"])
plot(p1, p2, p3, p4, legend, layout = @layout([[A B; C D] E{.1w}]))

