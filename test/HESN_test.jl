using DrWatson
@quickactivate "UCLCHEM Surrogate"

include(srcdir("EchoStateNetwork.jl"))

using DifferentialEquations, Plots, Sundials

using Random
Random.seed!(0)

include.(srcdir.(["GasPhaseNetwork.jl", "CVODESolve.jl", "Visualize.jl"])) #, "NetworkSurrogate.jl"]))

rfp, icfp, sfp = map(x -> datadir("exp_raw", x), ["reactions_final.csv", "initcond0.csv", "species.csv"])


tspan = (0., 10^6 * 365. * 24. * 3600.)
                      #  zeta, omega, T, F_UV, A_v, E, density
rates = [1e-17, 0.5, 10, 1., 10., 1e6]

par = Parameters(rates...)
prob = formulate_all(rfp, icfp, par, tspan=tspan)
p = ODEProblem(prob.network, prob.u0, prob.tspan)

dt = 3600. *24. *365. * 20
@time sol = solve(p,CVODE_BDF(), saveat=dt, abstol=10e-40, reltol=10e-15);
@time high_error_sol = solve(p,CVODE_BDF(), saveat=dt, abstol=10e-20, reltol=10e-15);

#@time sol = solve(prob, 3600. *24. *365. * 20, abstol=10e-30, reltol=10e-10);
X = hcat(sol.u...)
high_error_X = hcat(high_error_sol.u...)

X_train = X[:, begin:end-1]
y_train = X[:, begin+1:end]

warmup_length = 50
warmup = X_train[:, begin:warmup_length]

hesn = ESN.HybridEchoStateNetwork{Float64, ESN.EchoStateReservoir{Float64}}(234,250,234, p, CVODE_BDF, dt, abstol=10e-20, reltol=1e-15)
ESN.train!(hesn, [X_train[:, begin:warmup_length]], [y_train[:, begin:warmup_length]], 1e-11);

prediction = ESN.predict!(hesn, warmup, 2000);
println(Flux.Losses.mae(X[:, warmup_length:2000+warmup_length], prediction))
println(Flux.Losses.mae(X[:, warmup_length:2000+warmup_length], high_error_X[:, warmup_length:2000+warmup_length]))





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
                x->formulate_all(rfp, icfp, x, tspan=tspan) |>
                x->ODEProblem(x.network, x.u0, x.tspan)
            end

solutions = problems .|> x->@time  solve(x, CVODE_BDF(), abstol=10e-30, reltol=10e-8)