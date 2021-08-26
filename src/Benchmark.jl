using DrWatson
@quickactivate "UCLCHEM Surrogate"

using Surrogates, BenchmarkTools

include(srcdir("EchoStateNetwork.jl"))

using DifferentialEquations, Plots, Random

Random.seed!(0)

include.(srcdir.(["GasPhaseNetwork.jl", "CVODESolve.jl", "Visualize.jl"])) #, "NetworkSurrogate.jl"]))

rfp, icfp, sfp = map(x -> datadir("exp_raw", x), ["reactions_final.csv", "initcond0.csv", "species.csv"])
tspan = (0., 10^7 * 365. * 24. * 3600.)

rates_set_lower_bound = [1e-17, 0.5, 10, 1., 10., 1e2]
rates_set_upper_bound = [1e-14, 0.5, 300, 1., 10., 1e6]


function setup_problem(phys_params)
    physical_parameters = Parameters(phys_params...)
    cnp = formulate_all(rfp, icfp,physical_parameters, tspan=tspan)
end

parameter_samples = sample(15, rates_set_lower_bound, rates_set_upper_bound, SobolSample())
ps  = copy(parameter_samples)

bm = @benchmarkable solve(cnp, abstol=1e-20, reltol=1e-6) setup=(cnp = setup_problem($(pop!(ps)))) evals=1 seconds=500
t_low = run(bm)
violin(t_low.times .* 1e-9, label = "Low Tolerance", legend=:outertopright)

ps  = copy(parameter_samples)

bm = @benchmarkable solve(cnp, abstol=1e-25, reltol=1e-7) setup=(cnp = setup_problem($(pop!(ps)))) evals=1 seconds=500
t_med = run(bm)
violin!(t_med.times .* 1e-9, label = "Medium Tolerance", legend=:outertopright)

ps  = copy(parameter_samples)

bm = @benchmarkable solve(cnp, abstol=1e-30, reltol=1e-8) setup=(cnp = setup_problem($(pop!(ps)))) evals=1 seconds=500
t_high = run(bm)
violin!(t_high.times .* 1e-9, label = "High Tolerance", legend=:outertopright)

ps  = copy(parameter_samples)

bm = @benchmarkable solve(cnp, abstol=1e-40, reltol=1e-15)  setup=(cnp = setup_problem($(pop!(ps)))) evals=1 seconds=500
t_max = run(bm)
violin!(t_max.times .* 1e-9, label = "Max Tolerance", legend=:outertopright)

ylabel!("Time / seconds")
title!("CVODE BDF")
savefig(projectdir("images", "CVODE_benchmark.png"))

using Serialization
esn = deserialize(projectdir("bin", "esn"))
desn = deserialize(projectdir("bin", "desn"))


bm = @benchmarkable ESN.predict!(esn, data[:, 1:10], size(data, 2)-10) setup=(data = $(rand(ESN.inputdim(esn), 250))) evals=100 seconds=60 samples=100
d_250_esn = run(bm)

bm = @benchmarkable ESN.predict!(esn, data[:, 1:10], size(data, 2)-10) setup=(data = $(rand(ESN.inputdim(esn), 500))) evals=100 seconds=60 samples=100
d_500_esn = run(bm)

bm = @benchmarkable ESN.predict!(esn, data[:, 1:10], size(data, 2)-10) setup=(data = $(rand(ESN.inputdim(esn), 1000))) evals=100 seconds=60 samples=100
d_1000_esn = run(bm)

bm = @benchmarkable ESN.predict!(esn, data[:, 1:10], size(data, 2)-10) setup=(data = $(rand(ESN.inputdim(esn), 2500))) evals=5 seconds=600 samples=10
d_2500_esn = run(bm)

bm = @benchmarkable ESN.predict!(esn, data[:, 1:10], size(data, 2)-10) setup=(data = $(rand(ESN.inputdim(esn), 10000))) evals=5 seconds=600 samples=10
d_10000_esn = run(bm)
violin(d_250_esn.times .* 1e-9, label = "250 Steps", legend=:outertopright)
violin!(d_500_esn.times .* 1e-9, label = "500 Steps", legend=:outertopright)
violin!(d_1000_esn.times .* 1e-9, label = "1000 Steps", legend=:outertopright)
violin!(d_2500_esn.times .* 1e-9, label = "2500 Steps", legend=:outertopright)
violin!(d_10000_esn.times .* 1e-9, label = "10000 Steps", legend=:outertopright)
ylabel!("Time / seconds")
xlabel!("Steps")
xticks!([1,2,3,4,5], ["250", "500", "1000", "2500", "10000"])
title!("CTESN")
"""
Deep CTESN
"""
bm = @benchmarkable ESN.predict!(desn, data[:, 1:10], size(data, 2)-10) setup=(data = $(rand(ESN.inputdim(esn), 250))) evals=5 seconds=60 samples=100
d_250_desn = run(bm)
violin(d_250_desn.times .* 1e-9, label = "250 Steps", legend=:outertopright)

bm = @benchmarkable ESN.predict!(desn, data[:, 1:10], size(data, 2)-10) setup=(data = $(rand(ESN.inputdim(esn), 500))) evals=5 seconds=60 samples=100
d_500_desn = run(bm)
violin!(d_500_desn.times .* 1e-9, label = "500 Steps", legend=:outertopright)

bm = @benchmarkable ESN.predict!(desn, data[:, 1:10], size(data, 2)-10) setup=(data = $(rand(ESN.inputdim(esn), 1000))) evals=5 seconds=60 samples=100
d_1000_desn = run(bm)
violin!(d_1000_desn.times .* 1e-9, label = "1000 Steps", legend=:outertopright)

bm = @benchmarkable ESN.predict!(desn, data[:, 1:10], size(data, 2)-10) setup=(data = $(rand(ESN.inputdim(esn), 2500))) evals=5 seconds=600 samples=100
d_2500_desn = run(bm)
violin!(d_2500_desn.times .* 1e-9, label = "2500 Steps", legend=:outertopright)

bm = @benchmarkable ESN.predict!(desn, data[:, 1:10], size(data, 2)-10) setup=(data = $(rand(ESN.inputdim(esn), 10000))) evals=5 seconds=600 samples=100
d_10000_desn = run(bm)
violin!(d_10000_desn.times .* 1e-9, label = "10000 Steps", legend=:outertopright)
ylabel!("Time / seconds")