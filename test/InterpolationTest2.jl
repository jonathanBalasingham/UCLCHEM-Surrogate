using DrWatson
@quickactivate "UCLCHEM Surrogate"

using DifferentialEquations, Sundials, Serialization, ReservoirComputing, Surrogates, Plots

using Random
Random.seed!(0)

include.(srcdir.(["GasPhaseNetwork.jl", "CVODESolve.jl", "Visualize.jl"])) #, "NetworkSurrogate.jl"]))

rfp, icfp, sfp = map(x -> datadir("exp_raw", x), ["reactions_final.csv", "initcond0.csv", "species.csv"])

include(srcdir("EchoStateNetwork.jl"))
include(srcdir("Scoring.jl"))

tspan = (0., 10^7 * 365. * 24. * 3600.)
                      #  zeta, omega, T, F_UV, A_v, E, density
rates_set_lower_bound = [1e-17, 0.5, 10, 1., 10., 1e2]
rates_set_upper_bound = [1e-14, 0.5, 300, 1., 10., 1e6]
midpoint = (rates_set_lower_bound .+ rates_set_upper_bound) ./ 2
true_rates = get_rates(rfp, Parameters(midpoint...))

new_lower_bound = .95 .* true_rates
new_upper_bound = 1.05 .* true_rates

parameter_samples1 = sample(8, new_lower_bound, true_rates .* .981, SobolSample())
parameter_samples2 = sample(8, true_rates .* 1.016, new_upper_bound, SobolSample())
parameter_samples3 = sample(8, true_rates.* .961, true_rates .* 1.041, SobolSample())
parameter_samples = [parameter_samples1; parameter_samples2; parameter_samples3]

d = Dict()
#d = deserialize("weight_dict_esn")

function condition(u, t, integrator)
    check_error(integrator) != :Success
end

function affect!(integrator)
    terminate!(integrator)
end


callback = ContinuousCallback(condition, affect!)

pr = formulate_all(rfp, icfp, Parameters(zeros(6)...), tspan=tspan, rates=[parameter_samples[begin]...])
prob=ODEProblem(pr.network, pr.u0, pr.tspan)
@time sol = solve(prob, CVODE_BDF(), abstol=10e-20, reltol=10e-10, callback=callback)
train = hcat(sol.u...) |> x -> log2.(x .+ abs(minimum(x))*1.01)
X = [train[:, begin:end-1]]
y = [train[:, begin+1:end]]

timepoints = sol.t

warmup_length = 10
warmup = X[begin][:, begin:warmup_length]
steps = size(y[begin], 2) - size(warmup, 2)

input_dimesnion = size(X[begin], 1)
output_dimension = size(X[begin], 1)
esn = ESN.EchoStateNetwork{Float64, ESN.EchoStateReservoir{Float64}}(input_dimesnion, 1500, output_dimension);

function test!(esn, beta, X, y)
    ESN.train!(esn, X, y, beta)
    warmup_length = 10
    warmup = X[begin][:, begin:warmup_length]
    steps = size(y[begin], 2) - size(warmup, 2)
    pred = ESN.predict!(esn, warmup, steps) |> x->vcat(x, hcat(sum.(eachcol(2 .^ x[1:end, :]))...))
    _y = y[begin] |> x->vcat(x, hcat(sum.(eachcol(2 .^ x[1:end, :]))...))
    roc(pred, _y[begin:end, warmup_length:end])
  end
  
function test_all(esn, X, y, beta=20.0, reduction_factor=.6)
    error = Inf
    while true
    new_error = test!(esn, beta, X, y)
    if new_error > error || isinf(new_error)
        return (error, beta / reduction_factor)
    else
        error = new_error
    end
    beta *= reduction_factor
    end
end
  

esn = deserialize("./esn")
desn = deserialize("./desn")

err, beta = test_all(desn, X, y)

parameter_samples .|>
            begin
                x->formulate_all(rfp, icfp, Parameters(zeros(6)...), tspan=tspan, rates=[x...]) |>
                x->
                begin 
                    try
                        if (replace(log10.(x.rates .+ 1e-30), -Inf=>0.0)) in keys(d)
                            return
                        end
                        prob=ODEProblem(x.network, x.u0, x.tspan)
                        @time sol = solve(prob, CVODE_BDF(), abstol=10e-30, reltol=10e-15, callback=callback, saveat=timepoints)
                        if sol.t[end] >= tspan[2]*.999
                            train = hcat(sol.u...) |> x -> log2.(x .+ abs(minimum(x))*1.01)
                            X = [train[:, begin:end-1]]
                            y = [train[:, begin+1:end]]

                            err, beta = test_all(esn, X, y)
                            @info "Using beta: $beta with roc error: $err"
                            ESN.train!(esn, X, y, beta)
                            flattened_W_out = reshape(esn.output_layer.weight, :, 1)
                            d[replace(log10.(x.rates .+ 1e-30), -Inf=>0.0)] = flattened_W_out
                            @info "Weight dictionary has $(length(keys(d))) entries"
                        end
                    catch e
                        println(e)
                    end
                end
            end;



         

x_rates = keys(d)
lowerbound = hcat(x_rates...) |> eachrow .|> minimum
upperbound = hcat(x_rates...) |> eachrow .|> maximum
#weight_surrogate = RadialBasis(full .|> x->x[begin], full .|> x->x[end], replace(log10.(true_rates .+ 1e-30), -Inf=>0.0), replace(log10.(new_upper_bound .+ 1e-30), -Inf=>0.0))
weight_surrogate = RadialBasis([keys(d)...], [values(d)...], lowerbound, upperbound)

test_rates = parameter_samples[end] .* 1.02
test_parameters = [1e-15, 0.5, 10, 1., 10., 1e2]
pa = Parameters(test_parameters...)
p = formulate_all(rfp, icfp, pa, tspan=tspan, rates=[(true_rates)...])
@time sol = solve(ODEProblem(p.network, p.u0, p.tspan), CVODE_BDF(), abstol=10e-30, reltol=10e-15, saveat=timepoints)
train_subset = vcat(sol.t', hcat(sol.u...)) .|> x -> log2.(x .+ abs(minimum(x))*1.01)

buffer = 50

X = [train_subset[2:end, begin:end-1]]
y = [train_subset[2:end, begin+1:end]]

err, beta = test_all(desn, X, y)
W_out_dims = size(esn.output_layer.weight)

warmup_length = 10
warmup = X[begin][:, begin:warmup_length]
steps = size(y[begin], 2) - size(warmup, 2)


interp_rates = replace(log10.(p.rates .+ 1e-30), -Inf=>0.0)
W_out_interpolated = reshape(weight_surrogate(interp_rates), W_out_dims)
esn.output_layer.weight .= W_out_interpolated

prediction = ESN.predict!(esn, warmup, steps)

for i in 1:Integer(round(size(X[begin], 1) / 25))+1
    if i*25 > size(X[begin], 1)
      plot(10 .^ X_and_y[test_ind][rates_length+1, warmup_length+1:end],
        _y[(i-1)*25+1:end, warmup_length:end]',
        xscale=:log10,
        label="GT", layout=25, legend=nothing, size=(1200,800))
        plot!(10 .^ X_and_y[test_ind][rates_length+1, warmup_length+1:end], pred1[(i-1)*25+1:end, :]', xscale=:log10, layout=24)
        plot!(10 .^ X_and_y[test_ind][rates_length+1, warmup_length+1:end], pred2[(i-1)*25+1:end, :]', xscale=:log10, layout=24)      
    else
      plot(10 .^ X_and_y[test_ind][rates_length+1, warmup_length+1:end],
        _y[(i-1)*25+1:i*25, warmup_length:end]',
        xscale=:log10,
        label="GT", layout=25, legend=nothing, size=(1200,800))
        plot!(10 .^ X_and_y[test_ind][rates_length+1, warmup_length+1:end], pred1[(i-1)*25+1:i*25, :]', xscale=:log10, layout=24)
        plot!(10 .^ X_and_y[test_ind][rates_length+1, warmup_length+1:end], pred2[(i-1)*25+1:i*25, :]', xscale=:log10, layout=24)      
    end
    savefig(projectdir("test_plots", "Interpolation_ESR_species_$i.png"))
  end

# 6.692415508435203 -> 119
# 6.645032073561339 -> 143
# 6.6443856995188435 -> 167
# 6.653525239581264 -> 191