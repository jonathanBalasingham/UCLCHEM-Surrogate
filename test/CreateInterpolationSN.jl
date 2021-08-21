using DrWatson
@quickactivate "UCLCHEM Surrogate"

using DifferentialEquations, Sundials, Serialization, Surrogates, Plots

using Random
Random.seed!(0)

include.(srcdir.(["GasPhaseNetwork.jl", "CVODESolve.jl", "Visualize.jl"])) #, "NetworkSurrogate.jl"]))

rfp, icfp, sfp = map(x -> datadir("exp_raw", x), ["reactions_small.csv", "initcond0.csv", "species.csv"])

include(srcdir("EchoStateNetwork.jl"))
include(srcdir("Scoring.jl"))

tspan = (0., 10^7 * 365. * 24. * 3600.)
                      #  zeta, omega, T, F_UV, A_v, E, density
rates_set_lower_bound = [1e-17, 0.5, 10, 1., 10., 1e2]
rates_set_upper_bound = [1e-14, 0.5, 300, 1., 10., 1e6]

dark_cloud_upper = get_rates(rfp, Parameters(rates_set_upper_bound...))
dark_cloud_lower = get_rates(rfp, Parameters(rates_set_lower_bound...))
true_dark_cloud_lower = [min(a,b) for (a,b) in zip(dark_cloud_lower, dark_cloud_upper)]
true_dark_cloud_upper = [max(a,b) for (a,b) in zip(dark_cloud_lower, dark_cloud_upper)]

d = deserialize("weight_dict_esn")


train = simulation_dict |> values |> first
X = [train[:, begin:end-1]]
y = [train[:, begin+1:end]]

timepoints = deserialize("timepoints")

warmup_length = 10
warmup = X[begin][:, begin:warmup_length]
steps = size(y[begin], 2) - size(warmup, 2)

input_dimesnion = size(X[begin], 1)
output_dimension = size(y[begin], 1)
esn = ESN.DeepEchoStateNetwork{Float64, ESN.EchoStateReservoir{Float64}}(input_dimesnion, 30, 10, output_dimension);


transform(x; u0) = hcat((eachrow(x) ./ u0)...)' |> Matrix
transform_back(x; u0) = hcat((eachrow(x) .* u0)...)' |> Matrix
get_u0(x) = x[:, begin]

function test!(esn, beta, X, y; nrm=false)
    if nrm
        u0 = X[begin][:, begin]
        X_t = transform.(X, u0=u0)
        y_t = transform.(y, u0=u0)
        #ESN.train!(esn, X, y, beta)
        ESN.train!(esn, X_t, y_t, beta)
        warmup_length = 10
        warmup = X_t[begin][:, begin:warmup_length]
        steps = size(y[begin], 2) - size(warmup, 2)
        @time pred = ESN.predict!(esn, warmup, steps) # |> x->vcat(x, hcat(sum.(eachcol(2 .^ x[1:end, :]))...))
        pred = transform_back(pred, u0=u0)
        _y = y[begin] #|> x->vcat(x, hcat(sum.(eachcol(2 .^ x[1:end, :]))...))
        Flux.Losses.mae(pred, _y[begin:end, warmup_length:end])
    else
        ESN.train!(esn, X, y, beta)
        warmup_length = 10
        warmup = X[begin][:, begin:warmup_length]
        steps = size(y[begin], 2) - size(warmup, 2)
        @time pred = ESN.predict!(esn, warmup, steps) # |> x->vcat(x, hcat(sum.(eachcol(2 .^ x[1:end, :]))...))
        _y = y[begin] #|> x->vcat(x, hcat(sum.(eachcol(2 .^ x[1:end, :]))...))
        Flux.Losses.mae(pred, _y[begin:end, warmup_length:end])
    end
  end
  
function test_all(esn, X, y, beta=20.0, reduction_factor=.5; nrm=false)
    error = Inf
    while true
    new_error = test!(esn, beta, X, y, nrm=nrm)
    @info "error: $new_error"
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
bottom = filter(x->x>0,true_dark_cloud_lower) |> minimum |> log10 |> abs |> x->round(x)+1 
r(x) = replace(log10.(x) .+ bottom, -Inf=>0.0)


errors = Float64[]

for (rates, solution) in simulation_dict
    if rates in keys(esn_weight_dict)
        continue
    end
    X = [solution[:, begin:end-1]]
    y = [solution[:, begin+1:end]]
    u0 = get_u0(X[begin])
    err, beta = test_all(esn, X, y, nrm=true)
    push!(errors, err)
    @info "Using beta: $beta with mae error: $err"
    ESN.train!(esn, transform.(X, u0=u0), transform.(y, u0=u0), beta)
    flattened_W_out = reshape(esn.output_layer.weight, :, 1)
    esn_weight_dict[rates] = flattened_W_out
    @info "Weight dictionary has $(length(keys(esn_weight_dict))) entries"
    serialize("esn_weight_dict", esn_weight_dict)
end
      

weight_surrogate = RadialBasis([keys(esn_weight_dict)...], [values(esn_weight_dict)...], 
                                r(true_dark_cloud_lower), r(true_dark_cloud_upper))
#weight_surrogate = RadialBasis([keys(d)...], [values(d)...], lowerbound, upperbound)

test_parameters = [1e-16, 0.5, 230, 1., 10., 1e3]
test_rates = get_rates(rfp, Parameters(test_parameters...))
pa = Parameters(test_parameters...)
p = formulate_all(rfp, icfp, pa, tspan=tspan, rates=test_rates)
@time sol = solve(ODEProblem(p.network, p.u0, p.tspan), CVODE_BDF(), abstol=10e-30, reltol=10e-10, saveat=timepoints)
train_subset = vcat(sol.t', hcat(sol.u...)) .|> x -> log2.(x .+ abs(minimum(x))*1.01)


X = [train_subset[2:end, begin:end-1]]
y = [train_subset[2:end, begin+1:end]]
u0 = get_u0(X[begin])
err, beta = test_all(esn, X, y, 1e-1, nrm=true)
W_out_dims = size(esn.output_layer.weight)

warmup_length = 10
warmup = X[begin][:, begin:warmup_length]
steps = size(y[begin], 2) - size(warmup, 2)
ESN.train!(esn, transform.(X, u0 = u0), transform.(y, u0=u0), beta)
prediction2 = ESN.predict!(esn, transform(warmup, u0=u0), steps) |> x->transform_back(x, u0=u0)

interp_rates = r(p.rates)
W_out_interpolated = reshape(weight_surrogate(interp_rates), W_out_dims)
esn.output_layer.weight .= W_out_interpolated

prediction = ESN.predict!(esn, transform(warmup, u0=u0), steps) |> x->transform_back(x, u0=u0)

for i in 1:Integer(round(size(X[begin], 1) / 25))+1
    if i*25 > size(X[begin], 1)
      plot(timepoints[warmup_length+1:end],
        y[begin][(i-1)*25+1:end, warmup_length:end]',
        xscale=:log10,
        label="GT", layout=23, legend=nothing, size=(1200,800))
        plot!(timepoints[warmup_length+1:end], prediction[(i-1)*25+1:end, :]', xscale=:log10, layout=23)
        plot!(timepoints[warmup_length+1:end], prediction2[(i-1)*25+1:end, :]', xscale=:log10, layout=23)      
    else
        plot(timepoints[warmup_length+1:end],
        y[begin][(i-1)*25+1:i*25, warmup_length:end]',
        xscale=:log10,
        label="GT", layout=25, legend=nothing, size=(1200,800))
        #plot!(sol.t[warmup_length+1:end], prediction[(i-1)*25+1:i*25, :]', xscale=:log10, layout=25)
        plot!(sol.t[warmup_length+1:end], prediction2[(i-1)*25+1:i*25, :]', xscale=:log10, layout=23)    
    end
    savefig(projectdir("test_plots", "PeakSN_Final_Dark_Cloud_ESN_Interpolation_ESR_species_$i.png"))
  end

# 6.692415508435203 -> 119
# 6.645032073561339 -> 143
# 6.6443856995188435 -> 167
# 6.653525239581264 -> 191