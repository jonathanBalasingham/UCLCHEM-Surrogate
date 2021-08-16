using DrWatson
@quickactivate "UCLCHEM Surrogate"

using DifferentialEquations, Plots, Sundials, Serialization, ReservoirComputing, Surrogates

using Random
Random.seed!(0)

include.(srcdir.(["GasPhaseNetwork.jl", "CVODESolve.jl", "Visualize.jl"])) #, "NetworkSurrogate.jl"]))

rfp, icfp, sfp = map(x -> datadir("exp_raw", x), ["reactions_final.csv", "initcond0.csv", "species.csv"])


tspan = (0., 10^6 * 365. * 24. * 3600.)
                      #  zeta, omega, T, F_UV, A_v, E, density
rates_set_lower_bound = [1e-17, 0.5, 10, 1., 10., 1e2]
rates_set_upper_bound = [1e-14, 0.5, 300, 1., 10., 1e6]
midpoint = (rates_set_lower_bound .+ rates_set_upper_bound) ./ 2
true_rates = get_rates(rfp, Parameters(midpoint...))

new_lower_bound = .9 .* true_rates
new_upper_bound = 1.1 .* true_rates

parameter_samples1 = sample(50, new_lower_bound, true_rates, SobolSample())
parameter_samples2 = sample(50, true_rates, new_upper_bound, SobolSample())
parameter_samples = [parameter_samples1; parameter_samples2]

W = deserialize("models/reservoir")
W_in = deserialize("models/input_weights")
settings = deserialize("models/settings")

d = Dict()
full = parameter_samples .|>
            begin
                x->formulate_all(rfp, icfp, Parameters(zeros(6)...), tspan=tspan, rates=[x...]) |>
                x->
                begin 
                    try
                        prob=ODEProblem(x.network, x.u0, x.tspan)
                        @time sol = solve(prob, CVODE_BDF(), abstol=10e-30, reltol=10e-15)
                        if sol.t[end] >= tspan*.999
                            esn = ReservoirComputing.ESN(W, hcat(sol.u...) |> x -> log10.(x .+ abs(minimum(x))*1.01), W_in, alpha=settings.alpha, activation=tanh, nla_type=NLADefault(), extended_states=true)
                            W_out = ReservoirComputing.ESNtrain(esn, settings.beta)
                            flattened_W_out = reshape(W_out, :, 1)
                            #(replace(log10.(x.rates .+ 1e-30), -Inf=>0.0), flattened_W_out)
                            d[replace(log10.(x.rates .+ 1e-30), -Inf=>0.0)] = flattened_W_out
                        end
                    catch e
                        
                    end
                end
            end;


         

x_rates = full .|> x->x[begin]
lowerbound = hcat(x_rates...) |> eachrow .|> minimum
upperbound = hcat(x_rates...) |> eachrow .|> maximum
weight_surrogate = RadialBasis(full .|> x->x[begin], full .|> x->x[end], replace(log10.(true_rates .+ 1e-30), -Inf=>0.0), replace(log10.(new_upper_bound .+ 1e-30), -Inf=>0.0))
weight_surrogate = RadialBasis(full .|> x->x[begin], full .|> x->x[end], lowerbound, upperbound)

test_rates = parameter_samples[end] .* (1.02)
test_parameters = [1e-15, 0.5, 10, 1., 10., 1e2]
pa = Parameters(test_parameters...)
p = formulate_all(rfp, icfp, pa, tspan=tspan, rates=[test_rates...])
@time sol = solve(ODEProblem(p.network, p.u0, p.tspan), CVODE_BDF(), abstol=10e-30, reltol=10e-15)
train_subset = vcat(sol.t', hcat(sol.u...)) .|> x -> log10.(x .+ abs(minimum(x))*1.01)

buffer = 50

esn = ReservoirComputing.ESN(W, train_subset[2:end, :], W_in, alpha=settings.alpha, extended_states=true)
W_out = ESNtrain(esn, settings.beta)
esn = ReservoirComputing.ESN(W, train_subset[2:end, begin:buffer], W_in, alpha=settings.alpha, extended_states=true)
W_out_dims = size(W_out)

prediction2 = ESNpredict(esn, size(train_subset, 2)-buffer, W_out)

interp_rates = replace(log10.(get_rates(rfp, Parameters(test_parameters...))), -Inf=>0.0)
W_out_interpolated = reshape(weight_surrogate(interp_rates), W_out_dims)

esn = ReservoirComputing.ESN(W, train_subset[2:end, begin:buffer], W_in, alpha=settings.alpha, extended_states=true)

prediction = ESNpredict(esn, size(train_subset, 2)-buffer, W_out_interpolated)

for (i,k) in enumerate(eachrow(prediction))
    plot(train_subset[1, buffer+1:end], prediction[i, 1:end], title="", label="Interpolated", legend=:outertopright)
    plot!(train_subset[1,buffer+1:end],  prediction2[i, :], title="", label="Predicted", legend=:outertopright)
    plot!(train_subset[1,buffer+1:end],train_subset[i+1, buffer+1:end], title="", label="Groud Truth", legend=:outertopright)
    savefig(projectdir("plots_full_interp2","local_$(p.species[i]).png"))
end

