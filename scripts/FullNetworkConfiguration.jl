using DrWatson
@quickactivate "UCLCHEM Surrogate"

using Surrogates

include(srcdir("EchoStateNetwork.jl"))
include(srcdir("Scoring.jl"))

using DifferentialEquations, Plots, Sundials

using Random
Random.seed!(0)

include.(srcdir.(["GasPhaseNetwork.jl", "CVODESolve.jl", "Visualize.jl"])) #, "NetworkSurrogate.jl"]))

rfp, icfp, sfp = map(x -> datadir("exp_raw", x), ["reactions_final.csv", "initcond0.csv", "species.csv"])

function sample_around(n::Integer, rfp::String,physical_parameters::Parameters; method=SobolSample, above::Real=1.1, below::Real=.9)
  rates = get_rates(rfp, physical_parameters)
  Surrogates.sample(n, below .* rates, above .* rates, method()) .|> x->[x...]
end

tspan = (0., 10^7 * 365. * 24. * 3600.)
                      #  zeta, omega, T, F_UV, A_v, E, density
phys_params = [1e-14, 0.5, 300, 1., 10., 1e6]
rates_set = sample_around(5, rfp, Parameters(phys_params...))
@info "Solving sample set.."

#u0 = rand(23)
#u0 ./= sum(u0)
X_and_y = rates_set .|> 
            x->
            begin
              p = formulate_all(rfp, icfp, x, tspan=tspan)
              prob = ODEProblem(p.network, p.u0, p.tspan)
              sol = solve(prob, CVODE_BDF(), abstol=1e-25, reltol=1e-7)
              rates_repeated = reshape(repeat(x, length(sol.t)), length(x), :)
              vcat(rates_repeated, log10.(sol.t .+ 1e-30)', normalize_each_column(hcat(sol.u...)))
            end

rates_length = length(rates_set[begin])

#=

Here we declare data that will be used to find out regularization Parameter.
This parameter will be applied across all sets of data to determine error.

=#

X = X_and_y[begin:begin] .|> x->log2.(x[rates_length+2:end, begin:end-1])
y = X_and_y[begin:begin] .|> x->log2.(x[rates_length+2:end, begin+1:end])

input_dimension = size(X[begin], 1)
output_dimension = size(y[begin], 1)

test_ind = length(y)

warmup_length = 10
warmup = X[test_ind][:, begin:warmup_length]
steps = size(y[test_ind], 2) - size(warmup, 2)

_y = y[begin] |> x->vcat(x, hcat(sum.(eachcol(2 .^ x[1:end, :]))...))


function test!(esn, beta)
  ESN.train!(esn, X, y, beta)
  pred = ESN.predict!(esn, warmup, steps) |> x->vcat(x, hcat(sum.(eachcol(2 .^ x[1:end, :]))...))
  Flux.Losses.mae(pred, _y[begin:end, warmup_length:end])
end

function test_all(esn, beta=2.0, reduction_factor=.6)
    error = Inf
    while true
      new_error = test!(esn, beta)
      if new_error > error || isinf(new_error)
        return (error, beta / reduction_factor)
      else
        error = new_error
      end
      beta *= reduction_factor
    end
end



# Small Network, Single series static rates prediction
# MAE across thirty samples
X_set = X_and_y[begin+1:end] .|> x->log2.(x[rates_length+2:end, begin:end-1])
y_set = X_and_y[begin+1:end] .|> x->log2.(x[rates_length+2:end, begin+1:end])

@info "Starting ESR"
# ESR
esn = ESN.EchoStateNetwork{Float64, ESN.EchoStateReservoir{Float64}}(input_dimension,500,output_dimension);
desn = ESN.DeepEchoStateNetwork{Float64, ESN.EchoStateReservoir{Float64}}(input_dimension, 50, 10,output_dimension);

error_esr_ctesn = Float64[]
error_esr_deep_ctesn = Float64[]

err, esn_beta = test_all(esn, 100)
err, desn_beta = test_all(desn)

for (x_i,y_i) in zip(X_set, y_set)
  warmup_length_i = 10
  warmup_i = x_i[:, begin:warmup_length_i]
  steps_i = size(y_i, 2) - size(warmup_i, 2)

  ESN.train!(esn, [x_i], [y_i], esn_beta)
  ESN.train!(desn, [x_i], [y_i], desn_beta)
  pred1 = ESN.predict!(esn, warmup_i, steps_i)
  pred2 = ESN.predict!(desn, warmup_i, steps_i) 
  push!(error_esr_ctesn, Flux.Losses.mae(pred1, y_i[:, warmup_length_i:end]))
  push!(error_esr_deep_ctesn, Flux.Losses.mae(pred2, y_i[:, warmup_length_i:end]))
end

@info "Starting SCR"

# SCR
esn = ESN.EchoStateNetwork{Float64, ESN.SimpleCycleReservoir{Float64}}(input_dimension, 500, output_dimension; c=.1);
desn = ESN.DeepEchoStateNetwork{Float64, ESN.SimpleCycleReservoir{Float64}}(input_dimension, 50, 10, output_dimension, c=.1);

error_scr_ctesn = Float64[]
error_scr_deep_ctesn = Float64[]

err, esn_beta = test_all(esn, 100000.0)
err, desn_beta = test_all(desn, 3.4116345635915617e-6)


for (x_i,y_i) in zip(X_set, y_set)
  warmup_length_i = 10
  warmup_i = x_i[:, begin:warmup_length_i]
  steps_i = size(y_i, 2) - size(warmup_i, 2)

  ESN.train!(esn, [x_i], [y_i], esn_beta)
  ESN.train!(desn, [x_i], [y_i], desn_beta)
  pred1 = ESN.predict!(esn, warmup_i, steps_i)
  pred2 = ESN.predict!(desn, warmup_i, steps_i) 
  push!(error_scr_ctesn, Flux.Losses.mae(pred1, y_i[:, warmup_length_i:end]))
  push!(error_scr_deep_ctesn, Flux.Losses.mae(pred2, y_i[:, warmup_length_i:end]))
end

@info "Starting DLR"

# DLR
esn = ESN.EchoStateNetwork{Float64, ESN.DelayLineReservoir{Float64}}(input_dimension, 500, output_dimension; c=.7);
desn = ESN.DeepEchoStateNetwork{Float64, ESN.DelayLineReservoir{Float64}}(input_dimension, 50, 10, output_dimension, c=.7);

error_dlr_ctesn = Float64[]
error_dlr_deep_ctesn = Float64[]

err, esn_beta = test_all(esn, 100000)
err, desn_beta = test_all(desn, 26)

for (x_i,y_i) in zip(X_set, y_set)
  warmup_length_i = 10
  warmup_i = x_i[:, begin:warmup_length_i]
  steps_i = size(y_i, 2) - size(warmup_i, 2)

  ESN.train!(esn, [x_i], [y_i], esn_beta)
  ESN.train!(desn, [x_i], [y_i], desn_beta)
  pred1 = ESN.predict!(esn, warmup_i, steps_i)
  pred2 = ESN.predict!(desn, warmup_i, steps_i) 
  push!(error_dlr_ctesn, Flux.Losses.mae(pred1, y_i[:, warmup_length_i:end]))
  push!(error_dlr_deep_ctesn, Flux.Losses.mae(pred2, y_i[:, warmup_length_i:end]))
end

@info "Starting DLRF"

# DLRF
esn = ESN.EchoStateNetwork{Float64, ESN.DelayLineReservoir{Float64}}(input_dimension, 500, output_dimension, 0.1; c=.7, feedback=.2);
desn = ESN.DeepEchoStateNetwork{Float64, ESN.DelayLineReservoir{Float64}}(input_dimension, 50, 10, output_dimension, c=.7, feedback=.2);

error_dlrf_ctesn = Float64[]
error_dlrf_deep_ctesn = Float64[]

err, esn_beta = test_all(esn, 100)
err, desn_beta = test_all(desn, 1e-2)

for (x_i,y_i) in zip(X_set, y_set)
  warmup_length_i = 10
  warmup_i = x_i[:, begin:warmup_length_i]
  steps_i = size(y_i, 2) - size(warmup_i, 2)

  ESN.train!(esn, [x_i], [y_i], esn_beta)
  ESN.train!(desn, [x_i], [y_i], desn_beta)
  pred1 = ESN.predict!(esn, warmup_i, steps_i)
  pred2 = ESN.predict!(desn, warmup_i, steps_i) 
  push!(error_dlrf_ctesn, Flux.Losses.mae(pred1, y_i[:, warmup_length_i:end]))
  push!(error_dlrf_deep_ctesn, Flux.Losses.mae(pred2, y_i[:, warmup_length_i:end]))
end


using StatsPlots

errors = [error_esr_ctesn, error_esr_deep_ctesn, error_scr_ctesn, error_scr_deep_ctesn, 
          error_dlr_ctesn, error_dlr_deep_ctesn, error_dlrf_ctesn, error_dlrf_deep_ctesn]

boxplot(errors[1], label="ESR CTESN", yscale=:log10, legend=:outertopright)
boxplot!(errors[2], label="ESR Deep CTESN")
boxplot!(errors[3], label="SCR CTESN")
boxplot!(errors[4], label="SCR Deep CTESN")
boxplot!(errors[5], label="DLR CTESN")
boxplot!(errors[6], label="DLR Deep CTESN")
boxplot!(errors[7], label="DLRF CTESN")
boxplot!(errors[8], label="DLRF Deep CTESN")

savefig(projectdir("output", "MAE_Boxplots_FullNetwork.png"))

@info "Starting ROC Boxplots"

# ROC across the samples


function test!(esn, beta)
  ESN.train!(esn, X, y, beta)
  pred = ESN.predict!(esn, warmup, steps) |> x->vcat(x, hcat(sum.(eachcol(2 .^ x[1:end, :]))...))
  roc(pred, _y[begin:end, warmup_length:end])
end

function test_all(esn, beta=2.0, reduction_factor=.6)
    error = Inf
    while true
      new_error = test!(esn, beta)
      if new_error > error || isinf(new_error)
        return (error, beta / reduction_factor)
      else
        error = new_error
      end
      beta *= reduction_factor
    end
end

X_set = X_and_y[begin+1:end] .|> x->log2.(x[rates_length+2:end, begin:end-1])
y_set = X_and_y[begin+1:end] .|> x->log2.(x[rates_length+2:end, begin+1:end])

@info "Starting ESR"

# ESR
esn = ESN.EchoStateNetwork{Float64, ESN.EchoStateReservoir{Float64}}(input_dimension,500,output_dimension);
desn = ESN.DeepEchoStateNetwork{Float64, ESN.EchoStateReservoir{Float64}}(input_dimension, 50, 10,output_dimension);

error_esr_ctesn = Float64[]
error_esr_deep_ctesn = Float64[]

err, esn_beta = test_all(esn, 100)
err, desn_beta = test_all(desn, 5)

for (x_i,y_i) in zip(X_set, y_set)
  warmup_length_i = 10
  warmup_i = x_i[:, begin:warmup_length_i]
  steps_i = size(y_i, 2) - size(warmup_i, 2)

  ESN.train!(esn, [x_i], [y_i], esn_beta)
  ESN.train!(desn, [x_i], [y_i], desn_beta)
  pred1 = ESN.predict!(esn, warmup_i, steps_i)
  pred2 = ESN.predict!(desn, warmup_i, steps_i) 
  push!(error_esr_ctesn, roc(pred1, y_i[:, warmup_length_i:end]))
  push!(error_esr_deep_ctesn, roc(pred2, y_i[:, warmup_length_i:end]))
end

@info "Starting SCR"

# SCR
esn = ESN.EchoStateNetwork{Float64, ESN.SimpleCycleReservoir{Float64}}(input_dimension, 500, output_dimension; c=.1);
desn = ESN.DeepEchoStateNetwork{Float64, ESN.SimpleCycleReservoir{Float64}}(input_dimension, 50, 10, output_dimension, c=.1);

error_scr_ctesn = Float64[]
error_scr_deep_ctesn = Float64[]

err, esn_beta = test_all(esn, 100000)
err, desn_beta = test_all(desn)


for (x_i,y_i) in zip(X_set, y_set)
  warmup_length_i = 10
  warmup_i = x_i[:, begin:warmup_length_i]
  steps_i = size(y_i, 2) - size(warmup_i, 2)

  ESN.train!(esn, [x_i], [y_i], esn_beta)
  ESN.train!(desn, [x_i], [y_i], desn_beta)
  pred1 = ESN.predict!(esn, warmup_i, steps_i)
  pred2 = ESN.predict!(desn, warmup_i, steps_i) 
  push!(error_scr_ctesn, roc(pred1, y_i[:, warmup_length_i:end]))
  push!(error_scr_deep_ctesn, roc(pred2, y_i[:, warmup_length_i:end]))
end

@info "Starting DLR"

# DLR
esn = ESN.EchoStateNetwork{Float64, ESN.DelayLineReservoir{Float64}}(input_dimension, 500, output_dimension; c=.7);
desn = ESN.DeepEchoStateNetwork{Float64, ESN.DelayLineReservoir{Float64}}(input_dimension, 50, 10, output_dimension, c=.7);

error_dlr_ctesn = Float64[]
error_dlr_deep_ctesn = Float64[]

err, esn_beta = test_all(esn, 100000)
err, desn_beta = test_all(desn, 26)

for (x_i,y_i) in zip(X_set, y_set)
  warmup_length_i = 10
  warmup_i = x_i[:, begin:warmup_length_i]
  steps_i = size(y_i, 2) - size(warmup_i, 2)

  ESN.train!(esn, [x_i], [y_i], esn_beta)
  ESN.train!(desn, [x_i], [y_i], desn_beta)
  pred1 = ESN.predict!(esn, warmup_i, steps_i)
  pred2 = ESN.predict!(desn, warmup_i, steps_i) 
  push!(error_dlr_ctesn, roc(pred1, y_i[:, warmup_length_i:end]))
  push!(error_dlr_deep_ctesn, roc(pred2, y_i[:, warmup_length_i:end]))
end

@info "Starting DLRF"

# DLRF
esn = ESN.EchoStateNetwork{Float64, ESN.DelayLineReservoir{Float64}}(input_dimension, 500, output_dimension, 0.1; c=.7, feedback=.2);
desn = ESN.DeepEchoStateNetwork{Float64, ESN.DelayLineReservoir{Float64}}(input_dimension, 50, 10, output_dimension, c=.7, feedback=.2);

error_dlrf_ctesn = Float64[]
error_dlrf_deep_ctesn = Float64[]

err, esn_beta = test_all(esn, 100)
err, desn_beta = test_all(desn, 1e-2)

for (x_i,y_i) in zip(X_set, y_set)
  warmup_length_i = 10
  warmup_i = x_i[:, begin:warmup_length_i]
  steps_i = size(y_i, 2) - size(warmup_i, 2)

  ESN.train!(esn, [x_i], [y_i], esn_beta)
  ESN.train!(desn, [x_i], [y_i], desn_beta)
  pred1 = ESN.predict!(esn, warmup_i, steps_i)
  pred2 = ESN.predict!(desn, warmup_i, steps_i) 
  push!(error_dlrf_ctesn, roc(pred1, y_i[:, warmup_length_i:end]))
  push!(error_dlrf_deep_ctesn, roc(pred2, y_i[:, warmup_length_i:end]))
end


using StatsPlots

errors = [error_esr_ctesn, error_esr_deep_ctesn, error_scr_ctesn, error_scr_deep_ctesn, 
          error_dlr_ctesn, error_dlr_deep_ctesn, error_dlrf_ctesn, error_dlrf_deep_ctesn]

boxplot(errors[1], label="ESR CTESN", yscale=:log10, legend=:outertopright)
boxplot!(errors[2], label="ESR Deep CTESN")
boxplot!(errors[3], label="SCR CTESN")
boxplot!(errors[4], label="SCR Deep CTESN")
boxplot!(errors[5], label="DLR CTESN")
boxplot!(errors[6], label="DLR Deep CTESN")
boxplot!(errors[7], label="DLRF CTESN")
boxplot!(errors[8], label="DLRF Deep CTESN")
savefig(projectdir("output", "ROC_Boxplots_FullNetwork.png"))
