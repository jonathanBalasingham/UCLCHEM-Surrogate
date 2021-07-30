using Surrogates: Sobol
using Base: Float64
using DrWatson
@quickactivate "UCLCHEM Surrogate"

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
sol = solve(prob)
train = hcat(sol.u...)
rates_t = repeat(rates, length(sol.t)) |> x->reshape(x, length(rates), :)

X = train[:, 1:end-1]
y = train[:, 2:end]
esn = ESN.EchoStateNetwork{Float64}(3,500,3)
ESN.train!(esn, [X], [y], 1e-6)

startup = 70
plot(sol.t[2:startup], X[:, 2:startup]', xscale=:log10, label="ground truth", layout=3)
plot!(sol.t[2:startup], ESN.predict!(esn, X[:, 1:startup-1])', xscale=:log10, label="predicted", layout=3)

ESN.reset!(esn)

startup = 30
pred_length = length(sol.t) - startup - 1
prediction = ESN.predict!(esn, X[:, 1:startup])[:, end] # warmup the reservoir
for i in 1:pred_length
    prediction = hcat(prediction, esn(prediction[:, end]))
end

plot(sol.t[startup+1:end], y[:, startup:end]', xscale=:log10, label="ground truth", layout=3)
plot!(sol.t[startup+1:end], prediction', xscale=:log10, label="predicted", layout=3)

species_and_rates = vcat(rates_t, sol.t', train) |> x->log10.(x .+ 1e-30)
X = species_and_rates[:, 1:end-1]
y = species_and_rates[4:7, 2:end]

esn = ESN.EchoStateNetwork{Float64}(7, 500, 4)
ESN.train!(esn, [X], [y], 1e-4)

prediction = ESN.predict!(esn, X[:, 1:startup])[:, end] # warmup the reservoir
for i in 1:pred_length
    println(i)
    println(prediction)
    prediction = hcat(prediction, esn(vcat(X[1:3, i], prediction[:, end])))
end

plot(sol.t[startup+1:end], y[:, startup:end]', xscale=:log10, label="ground truth", layout=4)
plot!(sol.t[startup+1:end], prediction', xscale=:log10, label="predicted", layout=4)

using Surrogates, Sundials

rates_set1 = sample(50, [0.04, 0.01,0.01], [1.0, 4e5, 9e4],SobolSample())
rates_set2 = sample(50, [0.04, 0.01,0.01], [4e5, 1.0, 9e4],SobolSample())
rates_set3 = sample(50, [0.04, 0.01,0.01], [4e5, 4e5, 1.0],SobolSample())
rates_set = [rates_set1; rates_set2; rates_set3]
X = Matrix{Float64}[]
y = Matrix{Float64}[]

function solve_all(r)
    prob = ODEProblem(rober, u0, tspan, [r...])
    sol = solve(prob, Rosenbrock23())
    train = hcat(sol.u...)
    rates_t = repeat([r...], length(sol.t)) |> x->reshape(x, length(rates), :)

    species_and_rates = vcat(rates_t, sol.t', train)
    X_new = species_and_rates[:, 2:end-1]
    y_new = species_and_rates[4:7, 3:end]
    (X_new, y_new)
end

full = rates_set .|> solve_all 
X = full .|> (x->x[begin] .|> x->log10.(x))
y = full .|> (x->x[end] .|> x->log10.(x))

esn = ESN.EchoStateNetwork{Float64}(7, 800, 4);
@time ESN.train!(esn, X[begin:end-1], y[begin:end-1], 1.9)

test_ind = length(y)

prediction = ESN.predict!(esn, X[test_ind][:, 1:startup])[:, end] # warmup the reservoir
pred_length = size(y[test_ind], 2) - startup - 2
i = 1 # rate does not change
while prediction[begin, end] < y[test_ind][begin, end] && size(prediction, 2) < 500
    prediction = hcat(prediction, esn(vcat(X[test_ind][1:3, i], prediction[:, end])))
end

plot(y[test_ind][begin, startup:end], y[test_ind][begin+1:end, startup:end]', label="ground truth", layout=3, legend=:outertopright)
plot!(prediction[begin, :], prediction[begin+1:end, :]', label="predicted", layout=3)

# KNN
"""
esn = ESN.EchoStateNetwork{Float64}(7, 50, 4);

R = kmeans(hcat(X...), 50; maxiter=1000, display=:iter)
esn.input_layer.weight .= R.centers'

@time ESN.train!(esn, X[begin:end-1], y[begin:end-1], 1.5e-2)
test_ind = 5 #length(y)

prediction = ESN.predict!(esn, X[test_ind][:, 1:startup])[:, end] # warmup the reservoir
pred_length = size(y[test_ind], 2) - startup - 2
i = 1 # rate does not change
while prediction[begin, end] < y[test_ind][begin, end] && size(prediction, 2) < 500
    prediction = hcat(prediction, esn(vcat(X[test_ind][1:3, i], prediction[:, end])))
end

plot(y[test_ind][begin, startup:end], y[test_ind][begin+1:end, startup:end]', label="ground truth", layout=3, legend=:outertopright)
plot!(prediction[begin, :], prediction[begin+1:end, :]', label="predicted", layout=3)
"""

# Full Network

include.(srcdir.(["GasPhaseNetwork.jl", "CVODESolve.jl", "Visualize.jl"])) #, "NetworkSurrogate.jl"]))

rfp, icfp, sfp = map(x -> datadir("exp_raw", x), ["reactions_final.csv", "initcond4.csv", "species_final.csv"])

tspan = (0., 10^7 * 365. * 24. * 3600.)
                      #  zeta, omega, T, F_UV, A_v, E, density
rates_set_lower_bound = [1e-17, 0.5, 10, 1., 10., 1e2]
rates_set_upper_bound = [1e-14, 0.5, 300, 1., 10., 1e6]

parameter_samples = sample(30, rates_set_lower_bound, rates_set_upper_bound, SobolSample())

train = parameter_samples .|>
            begin
             x->Parameters(x...) |> 
             x->formulate_all(rfp, icfp, x) |>
             x->solve(x)
            end

# two possible pre-processes: each species between 0,1 
rates_length = length(train[begin].rates)

preprocess1(train_set::Matrix) = train_set[rates_length+2:end, :] |> x -> (x .- minimum(x)) ./ (maximum(x) - minimum(x))
preprocess2(train_set::Matrix) = eachcol(train_set) .|> x->x ./ sum(abs.(x))

full = train .|>
    sol->vcat(repeat([sol.rates...], length(sol.t)) |> x->reshape(x, length(sol.rates), :), sol.t', hcat(sol.u...)) 

full_preprocessed = (full .|> x->x[rates_length+2:end, :]) .= (full .|> (x-> preprocess1(x) |> preprocess2 |> x->hcat(x...)))

i = 1
for p in full_preprocessed
    full[i][rates_length+2:end,:] .= p
    i += 1
end

full .= full .|> x->log10.(x) |> x->replace(x, -Inf=>0.)
esn = ESN.EchoStateNetwork{Float64}(size(full[begin], 1), 3000, 235)

X = full .|> x->replace(x[:, begin:end-1], -Inf=>0.0)
y = full .|> x->replace(x[rates_length+1:end, 2:end], -Inf=>0.0)

@time ESN.train!(esn, X[begin:end-1], y[begin:end-1], 1.)

test_ind = length(y)

prediction = ESN.predict!(esn, X[test_ind][:, 1:startup])[:, end] # warmup the reservoir
pred_length = size(y[test_ind], 2) - startup - 2
i = 1 # rate does not change
while prediction[begin, end] < y[test_ind][begin, end] && size(prediction, 2) < 500
    prediction = hcat(prediction, esn(vcat(X[test_ind][1:3, i], prediction[:, end])))
end

plot(y[test_ind][begin, startup:end], y[test_ind][begin+1:end, startup:end]', label="ground truth", layout=3, legend=:outertopright)
plot!(prediction[begin, :], prediction[begin+1:end, :]', label="predicted", layout=3)
