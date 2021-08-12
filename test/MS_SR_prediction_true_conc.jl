


using Surrogates, Sundials

rates_set1 = sample(100, [0.04, 0.01,0.01], [1.0, 4e5, 9e4],SobolSample())
rates_set2 = sample(100, [0.04, 0.01,0.01], [4e5, 1.0, 9e4],SobolSample())
rates_set3 = sample(100, [0.04, 0.01,0.01], [4e5, 4e5, 1.0],SobolSample())
rates_set = [rates_set1; rates_set2; rates_set3]
X = Matrix{Float64}[]
y = Matrix{Float64}[]

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
X = full .|> x->x[begin]
y = full .|> x->x[end]

test_ind = length(y)

warmup_length = 10
warmup = X[test_ind][:, begin:warmup_length]
steps = size(y[test_ind], 2) - size(warmup, 2)

"""
Simple Test of fit on the Robertson Problem
"""

# ESR
esn = ESN.EchoStateNetwork{Float64, ESN.EchoStateReservoir{Float64}}(7,300,4);
desn = ESN.DeepEchoStateNetwork{Float64, ESN.EchoStateReservoir{Float64}}(7, 50, 6,4);

st = X[test_ind][begin:3, :]
xt = warmup[4:end, :]

ESN.train!(esn, X[1:end-1], y[1:end-1], 4e-3)
pred1 = ESN.predict!(esn, xt, st) |> x->vcat(x, hcat(sum.(eachcol(x[2:end, :]))...))


ESN.train!(desn, X[1:end-1], y[1:end-1], 5e-4)
pred2 = ESN.predict!(desn, xt, st) |> x->vcat(x, hcat(sum.(eachcol(x[2:end, :]))...))

_y = y[test_ind] |> x->vcat(x, hcat(sum.(eachcol(x[2:end, :]))...))

plot(10 .^ _y[begin, warmup_length:end], _y[2:end, warmup_length:end]', xscale=:log10, label="GT", layout=4, legend=:outertopright)
plot!(10 .^ pred1[begin, :], pred1[2:end, :]', xscale=:log10, label="CTESN", layout=4)
plot!(10 .^ pred2[begin, :], pred2[2:end, :]', xscale=:log10, label="DeepCTESN", layout=4)

savefig(projectdir("images", "MS_SR_TC_prediction_RP_ESR.png"))

# SCR
esn = ESN.EchoStateNetwork{Float64, ESN.SimpleCycleReservoir{Float64}}(7,600,4; c=.6);
desn = ESN.DeepEchoStateNetwork{Float64, ESN.SimpleCycleReservoir{Float64}}(7, 100, 6, 4, c=.7);

st = X[test_ind][begin:3, :]
xt = warmup[4:end, :]

ESN.train!(esn, X[1:end-1], y[1:end-1], 1e-6)
pred1 = ESN.predict!(esn, xt, st) |> x->vcat(x, hcat(sum.(eachcol(x[2:end, :]))...))


ESN.train!(desn, X[1:end-1], y[1:end-1], 1e-9)
pred2 = ESN.predict!(desn, xt, st) |> x->vcat(x, hcat(sum.(eachcol(x[2:end, :]))...))

_y = y[test_ind] |> x->vcat(x, hcat(sum.(eachcol(x[2:end, :]))...))

plot(10 .^ _y[begin, warmup_length:end], _y[2:end, warmup_length:end]', xscale=:log10, label="GT", layout=4, legend=:outertopright)
plot!(10 .^ pred1[begin, :], pred1[2:end, :]', xscale=:log10, label="CTESN", layout=4)
plot!(10 .^ pred2[begin, :], pred2[2:end, :]', xscale=:log10, label="DeepCTESN", layout=4)

savefig(projectdir("images", "MS_SR_TC_prediction_RP_SCR.png"))

# DLR
esn = ESN.EchoStateNetwork{Float64, ESN.DelayLineReservoir{Float64}}(7,300,4, c=.6);
desn = ESN.DeepEchoStateNetwork{Float64, ESN.DelayLineReservoir{Float64}}(7, 50, 6,4, c=.6);

ESN.train!(esn, [X], [y], 9e-5)
ESN.train!(desn, [X], [y], 5e-5)

st = X[begin:3, :]
xt = warmup[4:end, :]

pred1 = ESN.predict!(esn, xt, st) |> x->vcat(x, hcat(sum.(eachcol(x[2:end, :]))...))
pred2 = ESN.predict!(desn, xt, st) |> x->vcat(x, hcat(sum.(eachcol(x[2:end, :]))...))
_y = y |> x->vcat(x, hcat(sum.(eachcol(x[2:end, :]))...))

plot(sol.t[warmup_length+1:end], _y[2:end, warmup_length:end]', xscale=:log10, label="GT", layout=4, legend=:outertopright)
plot!(10 .^ pred1[begin, :], pred1[2:end, :]', xscale=:log10, label="CTESN", layout=4)
plot!(10 .^ pred2[begin, :], pred2[2:end, :]', xscale=:log10, label="DeepCTESN", layout=4)

savefig(projectdir("images", "MS_SR_TC_prediction_RP_DLR.png"))

# DLR w/F
esn = ESN.EchoStateNetwork{Float64, ESN.DelayLineReservoir{Float64}}(7,300,4, c=.6, feedback=.2);
desn = ESN.DeepEchoStateNetwork{Float64, ESN.DelayLineReservoir{Float64}}(7, 50, 6,4, c=.6, feedback=.3);

ESN.train!(esn, [X], [y], 8e-4)
ESN.train!(desn, [X], [y], 5e-5)

st = X[begin:3, :]
xt = warmup[4:end, :]

pred1 = ESN.predict!(esn, xt, st) |> x->vcat(x, hcat(sum.(eachcol(x[2:end, :]))...))
pred2 = ESN.predict!(desn, xt, st) |> x->vcat(x, hcat(sum.(eachcol(x[2:end, :]))...))
_y = y |> x->vcat(x, hcat(sum.(eachcol(x[2:end, :]))...))

plot(sol.t[warmup_length+1:end], _y[2:end, warmup_length:end]', xscale=:log10, label="GT", layout=4, legend=:outertopright)
plot!(10 .^ pred1[begin, :], pred1[2:end, :]', xscale=:log10, label="CTESN", layout=4)
plot!(10 .^ pred2[begin, :], pred2[2:end, :]', xscale=:log10, label="DeepCTESN", layout=4)

savefig(projectdir("images", "MS_SR_TC_prediction_RP_DLRF.png"))
