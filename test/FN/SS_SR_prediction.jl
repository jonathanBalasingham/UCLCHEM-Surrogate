using DrWatson
@quickactivate "UCLCHEM Surrogate"

using Surrogates

include(srcdir("EchoStateNetwork.jl"))
include(srcdir("Sample.jl"))
include(srcdir("Scoring.jl"))

using DifferentialEquations, Plots, Sundials

using Random
Random.seed!(0)

include.(srcdir.(["GasPhaseNetwork.jl", "CVODESolve.jl", "Visualize.jl"])) #, "NetworkSurrogate.jl"]))

rfp, icfp, sfp = map(x -> datadir("exp_raw", x), ["reactions_final.csv", "initcond0.csv", "species.csv"])


tspan = (0., 10^7 * 365. * 24. * 3600.)
                      #  zeta, omega, T, F_UV, A_v, E, density
phys_params = [1e-14, 0.5, 300, 1., 10., 1e6]
rates_set = sample_around(16, rfp, Parameters(phys_params...))

#u0 = rand(23)
#u0 ./= sum(u0)
X_and_y = rates_set .|> 
            x->
            begin
              p = formulate_all(rfp, icfp, Parameters(zeros(6)...), tspan=tspan, rates=x)
              prob = ODEProblem(p.network, p.u0, p.tspan)
              sol = solve(prob, CVODE_BDF(), abstol=1e-30, reltol=1e-10)
              rates_repeated = reshape(repeat(x, length(sol.t)), length(x), :)
              vcat(rates_repeated, log10.(sol.t .+ 1e-30)', normalize_each_column(hcat(sol.u...)))
            end

rates_length = length(rates_set[begin])

X = X_and_y[begin:begin] .|> x->log2.(x[rates_length+2:end, begin:end-1])
y = X_and_y[begin:begin] .|> x->log2.(x[rates_length+2:end, begin+1:end])

input_dimension = size(X[begin], 1)
output_dimension = size(y[begin], 1)

test_ind = length(y)

warmup_length = 10
warmup = X[test_ind][:, begin:warmup_length]
steps = size(y[test_ind], 2) - size(warmup, 2)

# not needed for input_dimesnion == output_dimension
#st = X[test_ind][begin:rates_length+1, :]
#xt = warmup[rates_length+2:end, :]


"""
Prediction on Single Systems of the full Network
"""
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

for i in 1:Integer(round(size(X[begin], 1) / 25))+1
  if i*25 > size(X[begin], 1)
    plot(10 .^ X_and_y[test_ind][rates_length+1, warmup_length+1:end],
      _y[(i-1)*25+1:end, warmup_length:end]',
      xscale=:log10,
      label="GT", layout=25, legend=:outertopright, size=(1200,800))
  else
    plot(10 .^ X_and_y[test_ind][rates_length+1, warmup_length+1:end],
      _y[(i-1)*25+1:i*25, warmup_length:end]',
      xscale=:log10,
      label="GT", layout=25, legend=:outertopright, size=(1200,800))
  end
  savefig(projectdir("test_plots", "species_$i.png"))
end

"""
Small Network, Single series static rates prediction
"""

"""
Echo State Reservoir 
"""
esn = ESN.EchoStateNetwork{Float64, ESN.EchoStateReservoir{Float64}}(input_dimension, 500, output_dimension);
desn = ESN.DeepEchoStateNetwork{Float64, ESN.EchoStateReservoir{Float64}}(input_dimension, 100, 5, output_dimension);
#sesn = ESN.SplitEchoStateNetwork{Float64, ESN. EchoStateReservoir{Float64}}((3,4), (200, 300), 4)

#ESN.train!(esn, X[1:end-1], y[1:end-1], 1.1372115211971872e-5)
#pred1 = ESN.predict!(esn, xt, st) |> x->vcat(x, hcat(sum.(eachcol(x[2:end, :]))...))

esn_error, esr_esn_beta = test_all(esn, 100)

ESN.train!(esn, X[end:end], y[end:end], esr_esn_beta)
pred1 = ESN.predict!(esn, warmup, steps) |> x->vcat(x, hcat(sum.(eachcol(2 .^ x[1:end, :]))...))

desn_error, esr_desn_beta = test_all(desn)

ESN.train!(esn, X[end:end], y[end:end], esr_desn_beta)
pred2 = ESN.predict!(desn, warmup, steps) |> x->vcat(x, hcat(sum.(eachcol(2 .^ x[1:end, :]))...))


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
  savefig(projectdir("test_plots", "ESR_species_$i.png"))
end

#

# SCR
esn = ESN.EchoStateNetwork{Float64, ESN.SimpleCycleReservoir{Float64}}(input_dimension, 500, output_dimension; c=.7);
desn = ESN.DeepEchoStateNetwork{Float64, ESN.SimpleCycleReservoir{Float64}}(input_dimension, 5, 100, output_dimension, c=.7);

esn_error, scr_esn_beta = test_all(esn, 100000.0)

ESN.train!(esn, X[end:end], y[end:end], scr_esn_beta)
pred1 = ESN.predict!(esn, warmup, steps) |> x->vcat(x, hcat(sum.(eachcol(2 .^ x[1:end, :]))...))

desn_error, scr_desn_beta = test_all(desn, 3.4116345635915617e-6)

ESN.train!(esn, X[end:end], y[end:end], scr_desn_beta)
pred2 = ESN.predict!(desn, warmup, steps) |> x->vcat(x, hcat(sum.(eachcol(2 .^ x[1:end, :]))...))

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
  savefig(projectdir("test_plots", "SCR_species_$i.png"))
end


# DLR
esn = ESN.EchoStateNetwork{Float64, ESN.DelayLineReservoir{Float64}}(input_dimension, 500, output_dimension; c=.7);
desn = ESN.DeepEchoStateNetwork{Float64, ESN.DelayLineReservoir{Float64}}(input_dimension, 5, 100, output_dimension, c=.7);

esn_error, dlr_esn_beta = test_all(esn, 100000.0)

ESN.train!(esn, X[end:end], y[end:end], dlr_esn_beta)
pred1 = ESN.predict!(esn, warmup, steps) |> x->vcat(x, hcat(sum.(eachcol(2 .^ x[1:end, :]))...))

desn_error, dlr_desn_beta = test_all(desn, 26)

ESN.train!(esn, X[end:end], y[end:end], dlr_desn_beta)
pred2 = ESN.predict!(desn, warmup, steps) |> x->vcat(x, hcat(sum.(eachcol(2 .^ x[1:end, :]))...))


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
  savefig(projectdir("test_plots", "DLR_species_$i.png"))
end


#DLRF
esn = ESN.EchoStateNetwork{Float64, ESN.DelayLineReservoir{Float64}}(input_dimension, 500, output_dimension, 0.1; c=.7, feedback=.2);
desn = ESN.DeepEchoStateNetwork{Float64, ESN.DelayLineReservoir{Float64}}(input_dimension, 5, 100, output_dimension, c=.7, feedback=.2);

esn_error, dlrf_esn_beta = test_all(esn, 100000.0)

ESN.train!(esn, X[end:end], y[end:end], esn_beta)
pred1 = ESN.predict!(esn, warmup, steps) |> x->vcat(x, hcat(sum.(eachcol(2 .^ x[1:end, :]))...))

desn_error, dlr_desn_beta = test_all(desn, 1e-3)

ESN.train!(esn, X[end:end], y[end:end], dlr_desn_beta)
pred2 = ESN.predict!(desn, warmup, steps) |> x->vcat(x, hcat(sum.(eachcol(2 .^ x[1:end, :]))...))

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
  savefig(projectdir("test_plots", "DLRF_species_$i.png"))
end

"""
MAE across thirty samples
"""
X_set = X_and_y[begin+1:end] .|> x->log2.(x[rates_length+2:end, begin:end-1])
y_set = X_and_y[begin+1:end] .|> x->log2.(x[rates_length+2:end, begin+1:end])


# ESR
esn = ESN.EchoStateNetwork{Float64, ESN.EchoStateReservoir{Float64}}(input_dimension,500,output_dimension);
desn = ESN.DeepEchoStateNetwork{Float64, ESN.EchoStateReservoir{Float64}}(input_dimension, 100, 5,output_dimension);

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

# SCR
esn = ESN.EchoStateNetwork{Float64, ESN.SimpleCycleReservoir{Float64}}(input_dimension, 500, output_dimension; c=.1);
desn = ESN.DeepEchoStateNetwork{Float64, ESN.SimpleCycleReservoir{Float64}}(input_dimension, 5, 100, output_dimension, c=.1);

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

# DLR
esn = ESN.EchoStateNetwork{Float64, ESN.DelayLineReservoir{Float64}}(input_dimension, 500, output_dimension; c=.7);
desn = ESN.DeepEchoStateNetwork{Float64, ESN.DelayLineReservoir{Float64}}(input_dimension, 5, 100, output_dimension, c=.7);

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

# DLRF
esn = ESN.EchoStateNetwork{Float64, ESN.DelayLineReservoir{Float64}}(input_dimension, 500, output_dimension, 0.1; c=.7, feedback=.2);
desn = ESN.DeepEchoStateNetwork{Float64, ESN.DelayLineReservoir{Float64}}(input_dimension, 5, 100, output_dimension, c=.7, feedback=.2);

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

savefig(projectdir("images", "Full_network_SS_SR_log2_box_plot_validation_15_MAE.png"))
"""
ROC across the samples
"""

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


# ESR
esn = ESN.EchoStateNetwork{Float64, ESN.EchoStateReservoir{Float64}}(input_dimension,500,output_dimension);
desn = ESN.DeepEchoStateNetwork{Float64, ESN.EchoStateReservoir{Float64}}(input_dimension, 100, 5,output_dimension);

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
  push!(error_esr_ctesn, roc(pred1, y_i[:, warmup_length_i:end]))
  push!(error_esr_deep_ctesn, roc(pred2, y_i[:, warmup_length_i:end]))
end

# SCR
esn = ESN.EchoStateNetwork{Float64, ESN.SimpleCycleReservoir{Float64}}(input_dimension, 500, output_dimension; c=.1);
desn = ESN.DeepEchoStateNetwork{Float64, ESN.SimpleCycleReservoir{Float64}}(input_dimension, 5, 100, output_dimension, c=.1);

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

# DLR
esn = ESN.EchoStateNetwork{Float64, ESN.DelayLineReservoir{Float64}}(input_dimension, 500, output_dimension; c=.7);
desn = ESN.DeepEchoStateNetwork{Float64, ESN.DelayLineReservoir{Float64}}(input_dimension, 5, 100, output_dimension, c=.7);

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

# DLRF
esn = ESN.EchoStateNetwork{Float64, ESN.DelayLineReservoir{Float64}}(input_dimension, 500, output_dimension, 0.1; c=.7, feedback=.2);
desn = ESN.DeepEchoStateNetwork{Float64, ESN.DelayLineReservoir{Float64}}(input_dimension, 5, 100, output_dimension, c=.7, feedback=.2);

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
savefig(projectdir("images", "Full_network_SS_SR_log2_box_plot_validation_15_ROC.png"))
