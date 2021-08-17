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

dt = 10.
rates = [0.04,3e7,1e4]
u0 = [1.0,1e-30,1e-30]
tspan = (0., 1e5)
prob = ODEProblem(rober, u0, tspan, rates)
sol = solve(prob, CVODE_BDF(), abstol=1e-16, reltol=1e-10, saveat=dt)
high_error_sol = solve(prob, CVODE_BDF(), abstol=1e-4, reltol=1e-2, saveat=dt)
train = hcat(sol.u...)
high_error_train = hcat(high_error_sol.u...)
rates_t = repeat(rates, length(sol.t)) |> x->reshape(x, length(rates), :)

X = train[:, 1:end-1]
high_error_X = high_error_train[:, 1:end-1]
y = train[:, 2:end]

"""
Simple Test of fit on the Robertson Problem
"""
# ESR
esn = ESN.EchoStateNetwork{Float64, ESN.EchoStateReservoir{Float64}}(3,300,3);
desn = ESN.DeepEchoStateNetwork{Float64, ESN.EchoStateReservoir{Float64}}(3, 50, 6,3);

# SCR
esn = ESN.EchoStateNetwork{Float64, ESN.SimpleCycleReservoir{Float64}}(3,300,3);
desn = ESN.DeepEchoStateNetwork{Float64, ESN.SimpleCycleReservoir{Float64}}(3, 50, 6,3);

# DLR
esn = ESN.EchoStateNetwork{Float64, ESN.DelayLineReservoir{Float64}}(3,300,3);
desn = ESN.DeepEchoStateNetwork{Float64, ESN.DelayLineReservoir{Float64}}(3, 50, 6,3);

# DLR w/F
esn = ESN.EchoStateNetwork{Float64, ESN.DelayLineReservoir{Float64}}(3,300,3);
desn = ESN.DeepEchoStateNetwork{Float64, ESN.DelayLineReservoir{Float64}}(3, 50, 6,3);


hesn = ESN.HybridEchoStateNetwork{Float64, ESN.EchoStateReservoir{Float64}}(3,250,3,prob, CVODE_BDF, dt, abstol=1e-6, reltol=1e-4)

ESN.train!(esn, [X], [y], 1e-4)
ESN.train!(desn, [X], [y], 1e-4)
ESN.train!(hesn, [X], [y], 1e-4)

startup = 70
plot(sol.t[2:end-1], X[:, 2:end]', xscale=:log10, label="GT", layout=3)
plot!(high_error_sol.t[2:end-1], high_error_X[:, 2:end]', xscale=:log10, label="HE GT", layout=3)
#plot!(sol.t[2:startup], ESN.predict!(esn, X[:, 1:startup-1])', xscale=:log10, label="CTESN", layout=3)
#plot!(sol.t[2:startup], ESN.predict!(desn, X[:, 1:startup-1])', xscale=:log10, label="DeepCTESN", layout=3)
plot!(high_error_sol.t[2:end-1], ESN.predict!(hesn, high_error_X[:, 1:end-1])', xscale=:log10, label="HESN", layout=3)

"""
Simple Prediction test w/ startup on the Robertson Problem 
"""
esn = ESN.EchoStateNetwork{Float64}(3,500,3);
desn = ESN.DeepEchoStateNetwork{Float64}(3, 50, 10, 3, .2, 1.1, .05);
ESN.train!(esn, [X], [y], 5e-4)
ESN.train!(desn, [X], [y], 9e-4)

startup = 30
pred_length = length(sol.t) - startup - 1
prediction = ESN.predict!(esn, X[:, 1:startup])[:, end] # warmup the reservoir
for i in 1:pred_length
    prediction = hcat(prediction, esn(prediction[:, end]))
end

prediction2 = ESN.predict!(desn, X[:, 1:startup])[:, end] # warmup the reservoir
for i in 1:pred_length
    prediction2 = hcat(prediction2, desn(prediction[:, end]))
end


startup = 30
pred_length = length(high_error_sol.t) - startup - 1
prediction3 = ESN.predict!(hesn, X[:, 1:startup])[:, end] # warmup the reservoir
while hesn.t[end] < tspan[2]
    prediction3 = hcat(prediction3, hesn(prediction3[:, end]))
end


plot(sol.t[2:end-1], X[:, 2:end]', xscale=:log10, label="GT", layout=3)
plot!(high_error_sol.t[2:end-1], high_error_X[:, 2:end]', xscale=:log10, label="HE GT", layout=3)
plot!(high_error_sol.t[2:end-1], prediction3', xscale=:log10, label="HE GT", layout=3)


plot(sol.t[startup+1:end], y[:, startup:end]', xscale=:log10, label="ground truth", layout=3)
plot!(sol.t[startup+1:end], prediction', xscale=:log10, label="CTESN", layout=3)
plot!(sol.t[startup+1:end], prediction2', xscale=:log10, label="DeepCTESN", layout=3)

plot(sol.t[2:end], y[:, startup:end]', xscale=:log10, label="ground truth", layout=3)
plot!(sol.t[startup+1:end], prediction', xscale=:log10, label="CTESN", layout=3)


"""
Prediction of Robertson Problem with Rates and time: "Master" Network
"""
species_and_rates = vcat(rates_t, sol.t', train) |> x->log10.(x .+ 1e-30)
X = species_and_rates[:, 1:end-1]
y = species_and_rates[4:7, 2:end]

esn = ESN.EchoStateNetwork{Float64, ESN.SimpleCycleReservoir{Float64}}(7,500,4;c=.7);
desn = ESN.DeepEchoStateNetwork{Float64, ESN.SimpleCycleReservoir{Float64}}(7, 50, 10, 4; c=.7);
ESN.train!(esn, [X], [y], 5e-3)
ESN.train!(desn, [X], [y], 7e-4)

startup = 30
pred_length = size(y, 2) - startup
prediction = ESN.predict!(esn, X[:, 1:startup])[:, end] # warmup the reservoir
for i in 1:pred_length
    prediction = hcat(prediction, esn(vcat(X[1:3, i], prediction[:, end])))
end

prediction2 = ESN.predict!(desn, X[:, 1:startup])[:, end] # warmup the reservoir
for i in 1:pred_length
    prediction2 = hcat(prediction2, desn(vcat(X[1:3, i], prediction2[:, end])))
end

plot(y[begin, startup:end], vcat(y[begin+1:end, startup:end], sum.(eachcol(10 .^ y[begin+1:end, startup:end]))')', label="ground truth", layout=4, legend=:outertopright)
plot!(prediction[begin, :], vcat(prediction[begin+1:end, :], sum.(eachcol(10 .^ prediction[begin+1:end, :]))')', label="CTESN", layout=3)
p = plot!(prediction2[begin, :], vcat(prediction2[begin+1:end, :], sum.(eachcol(10 .^ prediction2[begin+1:end, :]))')', label="DeepCTESN", layout=3)

"""

"""
using Surrogates, Sundials

rates_set1 = sample(200, [0.04, 0.01,0.01], [1.0, 4e5, 9e4],SobolSample())
rates_set2 = sample(200, [0.04, 0.01,0.01], [4e5, 1.0, 9e4],SobolSample())
rates_set3 = sample(200, [0.04, 0.01,0.01], [4e5, 4e5, 1.0],SobolSample())
rates_set = [rates_set1; rates_set2; rates_set3]
X = Matrix{Float64}[]
y = Matrix{Float64}[]

function solve_all(r)
    u = rand(3)
    u = u ./ sum(u)
    prob = ODEProblem(rober, u, tspan, [r...])
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

esn = ESN.EchoStateNetwork{Float64}(7, 1000, 4, .5, 1.0, .5);
desn = ESN.DeepEchoStateNetwork{Float64}(7, 100, 10, 4, .05, .1, .1, sparsity=0.7);

ESN.train!(esn, X, y, 5e-5)
ESN.train!(desn, X, y, 4e-3)

test_ind = rand(1:size(y,1))

prediction = ESN.predict!(esn, X[test_ind][:, 1:startup])[:, end] # warmup the reservoir
pred_length = size(y[test_ind], 2) - startup - 2
i = 1 # rate does not change
while prediction[begin, end] < y[test_ind][begin, end] && size(prediction, 2) < 500
    prediction = hcat(prediction, esn(vcat(X[test_ind][1:3, i], prediction[:, end])))
end

prediction2 = ESN.predict!(desn, X[test_ind][:, 1:startup])[:, end] # warmup the reservoir
pred_length = size(y[test_ind], 2) - startup - 2
i = 1 # rate does not change
while prediction2[begin, end] < y[test_ind][begin, end] && size(prediction2, 2) < 500
    prediction2 = hcat(prediction2, desn(vcat(X[test_ind][1:3, i], prediction2[:, end])))
end

pred1 = vcat(prediction[begin+1:end, :], sum.(eachcol(10 .^ prediction[begin+1:end, :]))')'
pred2 = vcat(prediction2[begin+1:end, :], sum.(eachcol(10 .^ prediction2[begin+1:end, :]))')'

plot(y[test_ind][begin, startup:end], vcat(y[test_ind][begin+1:end, startup:end], sum.(eachcol(10 .^ y[test_ind][begin+1:end, startup:end]))')', label="ground truth", layout=4, legend=:outertopright)
scatter!(prediction[begin, :], pred1, label="CTESN", layout=4)
p = scatter!(prediction2[begin, :], pred2, label="DeepCTESN", layout=4)

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

rfp, icfp, sfp = map(x -> datadir("exp_raw", x), ["reactions2.csv", "initcond0.csv", "species.csv"])

tspan = (0., 10^6 * 365. * 24. * 3600.)
                      #  zeta, omega, T, F_UV, A_v, E, density
rates_set_lower_bound = [1e-17, 0.5, 10, 1., 10., 1e2]
rates_set_upper_bound = [9.9e-17, 0.5, 300, 1., 10., 1e6]

parameter_samples1 = sample(10, rates_set_lower_bound, rates_set_upper_bound, SobolSample())

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

X = full .|> x->replace(x[:, begin:end-1], -Inf=>0.0)
y = full .|> x->replace(x[rates_length+1:end, 2:end], -Inf=>0.0)

Random.seed!(0)
function test_small_network(alpha, beta, sigma, radius)
    esn = ESN.EchoStateNetwork{Float64}(size(X[begin],1), 600, length(train[begin].species) + 1, sigma, radius, alpha);

    ESN.train!(esn, X[begin:end-1], y[begin:end-1], beta)

    test_ind = length(y)
    rates_length = train[begin].rates |> length

    @time prediction = ESN.predict!(esn, X[test_ind][:, 1:startup])[:, end] # warmup the reservoir
    pred_length = size(y[test_ind], 2) - startup - 2
    i = 1 # rate does not change
    while prediction[begin, end] < y[test_ind][begin, end] && size(prediction, 2) < 5000
        prediction = hcat(prediction, esn(vcat(X[test_ind][begin:rates_length, i], prediction[:, end])))
    end

    if prediction[begin, end] < 0 || any(x->x <= -5, prediction[begin, :])
        return
    end

    plot(y[test_ind][begin, startup:end], y[test_ind][begin+1:17, startup:end]', label="GT", layout=16, legend=:outertopright, size=(1400, 1000))
    p1 = scatter!(prediction[begin, :], prediction[2:17, :]', label="CTESN", layout=17)
    println("$(alpha)_$(beta)_$(sigma)_$(radius)")
    savefig(p1, projectdir("Small_network", "$(alpha)_$(beta)_$(sigma)_$(radius)_p1.png"))

    plot(y[test_ind][begin, startup:end], y[test_ind][18:end, startup:end]', label="GT", layout=17, legend=:outertopright, size=(1400, 1000))
    p2=scatter!(prediction[begin, :], prediction[18:end, :]', label="CTESN", layout=17)
    savefig(p2, projectdir("Small_network", "$(alpha)_$(beta)_$(sigma)_$(radius)_p2.png"))
end

#desn = ESN.DeepEchoStateNetwork{Float64}(size(X[begin],1), 100, 10, length(train[begin].species) + 1, .05, .1, .1, sparsity=0.7);
#ESN.train!(desn, X[begin:end-1], y[begin:end-1], 4e-3)

for a in .1:.2:1.0
    for b in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
        for s in [.1, .5, 1., 2]
            for r in [.7,.8,.9,1.0]
                test_small_network(a,b,s,r)
            end
        end
    end
end


function test_small_network_deep(alpha, beta, sigma, radius)
    desn = ESN.DeepEchoStateNetwork{Float64}(size(X[begin],1), 100, 10, length(train[begin].species) + 1, sigma, radius, alpha);

    ESN.train!(desn, X[begin:end-1], y[begin:end-1], beta)

    test_ind = length(y)
    rates_length = train[begin].rates |> length

    @time prediction = ESN.predict!(desn, X[test_ind][:, 1:startup])[:, end] # warmup the reservoir
    pred_length = size(y[test_ind], 2) - startup - 2
    i = 1 # rate does not change
    while prediction[begin, end] < y[test_ind][begin, end] && size(prediction, 2) < 5000
        prediction = hcat(prediction, desn(vcat(X[test_ind][begin:rates_length, i], prediction[:, end])))
    end

    if prediction[begin, end] < 0 || any(x->x <= -5, prediction[begin, :]) || any(x->x < -100, prediction)
        return
    end

    plot(y[test_ind][begin, startup:end], y[test_ind][begin+1:17, startup:end]', label="GT", layout=16, legend=:outertopright, size=(1400, 1000))
    p1 = scatter!(prediction[begin, :], prediction[2:17, :]', label="CTESN", layout=17)
    println("$(alpha)_$(beta)_$(sigma)_$(radius)")
    savefig(p1, projectdir("Small_network", "deep_$(alpha)_$(beta)_$(sigma)_$(radius)_p1.png"))

    plot(y[test_ind][begin, startup:end], y[test_ind][18:end, startup:end]', label="GT", layout=17, legend=:outertopright, size=(1400, 1000))
    p2=scatter!(prediction[begin, :], prediction[18:end, :]', label="CTESN", layout=17)
    savefig(p2, projectdir("Small_network", "deep_$(alpha)_$(beta)_$(sigma)_$(radius)_p2.png"))
end

for a in .4:.1:1.0
    for b in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
        for s in [.1, .5, 1., 2]
            for r in [.7,.8,.9,1.0]
                test_small_network(a,b,s,r)
            end
        end
    end
end