using Surrogates: Sobol
using DrWatson
@quickactivate "UCLCHEM Surrogate"

include(srcdir("Scoring.jl"))

include(srcdir("EchoStateNetwork.jl"))


using DifferentialEquations, Plots

include.(srcdir.(["GasPhaseNetwork.jl", "CVODESolve.jl", "Visualize.jl"])) #, "NetworkSurrogate.jl"]))

rfp, icfp, sfp = map(x -> datadir("exp_raw", x), ["reactions2.csv", "initcond0.csv", "species.csv"])


tspan = (0., 10^7 * 365. * 24. * 3600.)
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

problems = parameter_samples .|>
            begin
                x->Parameters(x...) |> 
                x->formulate_all(rfp, icfp, x)
            end

train = problems |>
             x->solve(x)

# two possible pre-processes: each species between 0,1 
rates_length = length(train[begin].rates)

preprocess1(train_set::Matrix) = train_set[rates_length+2:end, :] |> x -> (x .- minimum(x)) ./ (maximum(x) - minimum(x))
preprocess2(train_set::Matrix) = eachcol(train_set) .|> x->x ./ sum(abs.(x))
