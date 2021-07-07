using ReservoirComputing, Sundials, Surrogates

#= 

For Tomorrow:
    - add species to interpolation
    - add various transformation to interpolation
    - check other interpolation methods
    - with/without time ESN comparison
    - CTESN to ESN


=#

rfp = "./test/input/reactions_final.csv"
sfp = "./test/input/species_final.csv"
icfp = "./test/input/initcond4.csv"

#rfp = "./test/input/reactions.csv"
#sfp = "./test/input/species.csv"
#icfp = "./test/input/initcond1.csv"


function create_sample(lower_bound, upper_bound, n_samples; sample_type=SobolSample)
    sample(n_samples, lower_bound, upper_bound, sample_type())
end

rates_set_lower_bound = [1e-17, 0.5, 10, 0.5, 2.,  1e2]
rates_set_upper_bound = [1., 0.5, 100, 1.5, 10.,  1e4]


parameter_samples = sample(30, rates_set_lower_bound, rates_set_upper_bound, SobolSample())


res_size = 1500
radius = .8
degree = 1200
activation = tanh
alpha = .9
sigma = .1
nla_type = NLADefault()
extended_states = false
beta = 0.000001

resulting_weights = []
resulting_timestamps = Float64[]
parameter_samples_with_t = []



i = 1
parameter_sample = parameter_samples[begin]
pa = Parameters(parameter_sample...)
#p = UCLCHEM.formulate(sfp,rfp,icfp,pa,tspan, rate_factor = 1)
p = formulate_all(rfp, icfp, pa)
sol = solve(p, solver=CVODE_BDF)
saveat = sol.t

train_subset = hcat(sol.u...) |> Matrix
#train_subset = vcat(sol.t', hcat(sol.u...)) .|> tanh

#train_subset = vcat(sol.t', hcat(sol.u...)) .+ 1 .|> log10

#train_subset = vcat( sol.t' ./ tspan[2], (hcat(sol.u...) |> Matrix))
#train_subset = vcat( log10.(sol.t)', (hcat(sol.u...) |> Matrix))
#train_subset = vcat( log10.(sol.t)' ./ log10(tspan[end]), (hcat(sol.u...) |> Matrix))

#train_subset[train_subset .<= 0.0] .= 1e-60resulting_weights

esn = ESN(res_size,
        train_subset,
        degree,
        radius,
        activation = activation,
        alpha = alpha, 
        sigma = sigma, 
        nla_type = nla_type, 
        extended_states = extended_states)

W = esn.W
W_in = esn.W_in
successful_indx = zeros(Integer, length(parameter_samples))

for parameters in parameter_samples
    try
        println(i)
        pa = Parameters(parameters...)
        p = formulate_all(rfp, icfp, pa)
        #p = UCLCHEM.formulate(sfp,rfp,icfp,pa,tspan, rate_factor = 1)
        @time sol = solve(p, saveat, solver=CVODE_BDF);   
        train_subset = hcat(sol.u...) |> Matrix

        #train_subset = vcat( log10.(sol.t)', (hcat(sol.u...) |> Matrix))
        #train_subset = vcat( log10.(sol.t)' ./ log10(tspan[end]), (hcat(sol.u...) |> Matrix))

        esn = ESN(W,
                train_subset,
                W_in,
                activation = activation,
                alpha = alpha, 
                nla_type = nla_type, 
                extended_states = extended_states)
    
        @time W_out = ESNtrain(esn, beta)
        flattened_W_out = reshape(W_out, :, 1)
        push!(resulting_weights, flattened_W_out)
        for i in 1:length(sol.t) - 1
            params_and_t = [parameters..., log10(sol.t[i])]
            push!(parameter_samples_with_t, tuple(params_and_t...))
            push!(resulting_timestamps, log10(sol.t[i+1]))
        end
 
        successful_indx[i] = i   
        i += 1 
    catch e
        println("Failed with parameters: $pa")
        println(e)
    end  
end


weight_surrogate = RadialBasis(parameter_samples, resulting_weights, rates_set_lower_bound, rates_set_upper_bound)
timestamp_surrogate = SecondOrderPolynomialSurrogate(parameter_samples_with_t .|> x -> convert(NTuple{8, Float64}, x),
                               map(Float64, resulting_timestamps),
                               [rates_set_lower_bound; 0.0],
                               [rates_set_upper_bound; log10(tspan[2])])

test_parameters = rates_set_lower_bound  .+ ((rates_set_upper_bound .- rates_set_lower_bound) .* .5)
test_W_out = reshape(weight_surrogate(test_parameters), length(p.species), :)
test_t = timestamp_surrogate([test_parameters; 0.0])

pa = Parameters(test_parameters...)
#p = UCLCHEM.formulate(sfp,rfp,icfp,pa,tspan, rate_factor = 1)
p = formulate_all(rfp, icfp, pa)
sol = solve(p, saveat, solver=CVODE_BDF)

#train_subset = vcat( log10.(sol.t)' ./ log10(tspan[end]), (hcat(sol.u...) |> Matrix) )
train_subset = hcat(sol.u...) |> Matrix


esn = ESN(W,
        train_subset,
        W_in,
        activation = activation,
        alpha = alpha, 
        nla_type = nla_type, 
        extended_states = extended_states)

@time W_out = ESNtrain(esn, beta)

@time test_output = ESNfitted(esn, test_W_out)
@time output = ESNfitted(esn, W_out)

#plot(output[1:20, :]',layout=(4,5), label="Predicted", legend=:outertopright, size=(1800, 800))
#title!(string(test_parameters))

#plot!(test_output[1:20, :]',layout=(4,5), label="Interpolated", size=(1800, 1200))

#plot!(train_subset[1:20, 2:end]', layout=(4,5), label="Ground Truth", size=(1800, 1200))
#xaxis!(:log10)

#output[output .<= 0.0] .= 1e-60
#train_subset[train_subset .<= 0.0] .= 1e-60
#test_output[test_output .<= 0.0] .= 1e-60

#l = @layout [a{0.01h}; grid(2,2)]
#[plots[i+1] = plot(sol.t,train_subset[i, :],framestyle=:default,label="ground truth",title=p.species[i]) for i in 1:length(p.species)]
#plots = [plot(sol.t ./ (3600 * 24 * 365),train_subset[i, :],label="ground truth",title=p.species[i], size=(300,300), yaxis=:log10, xaxis=:log10) for i in 1:length(p.species)]
#plot(plots..., size=(1200,1000))


res_size = 1500
radius = .8
degree = 1200
activation = tanh
alpha = .9
sigma = .1
nla_type = NLADefault()
extended_states = false
beta = 0.000001

resulting_weights = []
resulting_timestamps = Float64[]
parameter_samples_with_t = []



i = 1
parameter_sample = parameter_samples[begin]
pa = Parameters(parameter_sample...)
#p = UCLCHEM.formulate(sfp,rfp,icfp,pa,tspan, rate_factor = 1)
p = formulate_all(rfp, icfp, pa)
sol = solve(p, solver=CVODE_BDF)
saveat = sol.t

train_subset = hcat(sol.u...) |> Matrix
#train_subset = vcat(sol.t', hcat(sol.u...)) .|> tanh

#train_subset = vcat(sol.t', hcat(sol.u...)) .+ 1 .|> log10

#train_subset = vcat( sol.t' ./ tspan[2], (hcat(sol.u...) |> Matrix))
#train_subset = vcat( log10.(sol.t)', (hcat(sol.u...) |> Matrix))
#train_subset = vcat( log10.(sol.t)' ./ log10(tspan[end]), (hcat(sol.u...) |> Matrix))

#train_subset[train_subset .<= 0.0] .= 1e-60resulting_weights

esn = ESN(res_size,
        train_subset,
        degree,
        radius,
        activation = activation,
        alpha = alpha, 
        sigma = sigma, 
        nla_type = nla_type, 
        extended_states = extended_states)

W = esn.W
W_in = esn.W_in
successful_indx = zeros(Integer, length(parameter_samples))

for parameters in parameter_samples
    try
        println(i)
        pa = Parameters(parameters...)
        p = formulate_all(rfp, icfp, pa)
        #p = UCLCHEM.formulate(sfp,rfp,icfp,pa,tspan, rate_factor = 1)
        @time sol = solve(p, saveat, solver=CVODE_BDF);   
        train_subset = hcat(sol.u...) |> Matrix

        #train_subset = vcat( log10.(sol.t)', (hcat(sol.u...) |> Matrix))
        #train_subset = vcat( log10.(sol.t)' ./ log10(tspan[end]), (hcat(sol.u...) |> Matrix))

        esn = ESN(W,
                train_subset,
                W_in,
                activation = activation,
                alpha = alpha, 
                nla_type = nla_type, 
                extended_states = extended_states)
    
        @time W_out = ESNtrain(esn, beta)
        flattened_W_out = reshape(W_out, :, 1)
        push!(resulting_weights, flattened_W_out)
        for i in 1:length(sol.t) - 1
            params_and_t = [parameters..., log10(sol.t[i])]
            push!(parameter_samples_with_t, tuple(params_and_t...))
            push!(resulting_timestamps, log10(sol.t[i+1]))
        end
 
        successful_indx[i] = i   
        i += 1 
    catch e
        println("Failed with parameters: $pa")
        println(e)
    end  
end


weight_surrogate = RadialBasis(parameter_samples, resulting_weights, rates_set_lower_bound, rates_set_upper_bound)
timestamp_surrogate = SecondOrderPolynomialSurrogate(parameter_samples_with_t .|> x -> convert(NTuple{8, Float64}, x),
                               map(Float64, resulting_timestamps),
                               [rates_set_lower_bound; 0.0],
                               [rates_set_upper_bound; log10(tspan[2])])

test_parameters = rates_set_lower_bound  .+ ((rates_set_upper_bound .- rates_set_lower_bound) .* .5)
test_W_out = reshape(weight_surrogate(test_parameters), length(p.species), :)
test_t = timestamp_surrogate([test_parameters; 0.0])

pa = Parameters(test_parameters...)
#p = UCLCHEM.formulate(sfp,rfp,icfp,pa,tspan, rate_factor = 1)
p = formulate_all(rfp, icfp, pa)
sol = solve(p, saveat, solver=CVODE_BDF)

#train_subset = vcat( log10.(sol.t)' ./ log10(tspan[end]), (hcat(sol.u...) |> Matrix) )
train_subset = hcat(sol.u...) |> Matrix


esn = ESN(W,
        train_subset,
        W_in,
        activation = activation,
        alpha = alpha, 
        nla_type = nla_type, 
        extended_states = extended_states)

@time W_out = ESNtrain(esn, beta)

@time test_output = ESNfitted(esn, test_W_out)
@time output = ESNfitted(esn, W_out)

#plot(output[1:20, :]',layout=(4,5), label="Predicted", legend=:outertopright, size=(1800, 800))
#title!(string(test_parameters))

#plot!(test_output[1:20, :]',layout=(4,5), label="Interpolated", size=(1800, 1200))

#plot!(train_subset[1:20, 2:end]', layout=(4,5), label="Ground Truth", size=(1800, 1200))
#xaxis!(:log10)

#output[output .<= 0.0] .= 1e-60
#train_subset[train_subset .<= 0.0] .= 1e-60
#test_output[test_output .<= 0.0] .= 1e-60

#l = @layout [a{0.01h}; grid(2,2)]
#[plots[i+1] = plot(sol.t,train_subset[i, :],framestyle=:default,label="ground truth",title=p.species[i]) for i in 1:length(p.species)]
#plots = [plot(sol.t ./ (3600 * 24 * 365),train_subset[i, :],label="ground truth",title=p.species[i], size=(300,300), yaxis=:log10, xaxis=:log10) for i in 1:length(p.species)]
#plot(plots..., size=(1200,1000))


using Plots
for (i, species) in enumerate(p.species)
    plot(sol.t ./ (3600 * 24 * 365), output[i, :], title=species, label="Predicted", legend=:outertopright)
    plot!(sol.t ./ (3600 * 24 * 365), train_subset[i, :], title=species, label="Groud Truth", legend=:outertopright)
    plot!(sol.t ./ (3600 * 24 * 365), test_output[i, :], title=species, label="Interpolated", legend=:outertopright)
    xticks!([10^0, 10,10^2,10^3,10^4,10^5,10^6])
    xaxis!(:log10)
    xlims!((1., 10^7))
    ylims!((10^-20, 1))
    yaxis!(:log10)
    savefig("./plots/$species.png")
end

using Plots
for (i, species) in enumerate(p.species)
    plot(sol.t ./ (3600 * 24 * 365), output[i, :], title=species, label="Predicted", legend=:outertopright)
    plot!(sol.t ./ (3600 * 24 * 365), train_subset[i, :], title=species, label="Groud Truth", legend=:outertopright)
    plot!(sol.t ./ (3600 * 24 * 365), test_output[i, :], title=species, label="Interpolated", legend=:outertopright)
    xticks!([10^0, 10,10^2,10^3,10^4,10^5,10^6])
    xaxis!(:log10)
    xlims!((1., 10^7))
    ylims!((10^-20, 1))
    yaxis!(:log10)
    savefig("./plots/$species.png")
end

# when predicting with log10 time
using Plots
output_t = sort(10 .^ output[1,:]) .- 1
test_t = sort(10 .^ test_output[1, :]) .- 1

output_t = 10 .^ (output[1,:] .* log10(tspan[end]))
test_t = 10 .^ (test_output[1, :] .* log10(tspan[end]))

plot_output = 10 .^ output .- 1
plot_train_subset = 10 .^ train_subset .- 1
plot_test_output = 10 .^ test_output .- 1

for (i, species) in enumerate(p.species)
    plot(output_t ./ (3600 * 24 * 365), plot_output[i+1, :], title=species, label="Predicted", legend=:outertopright)
    plot!(sol.t ./ (3600 * 24 * 365), plot_train_subset[i+1, :], title=species, label="Groud Truth", legend=:outertopright)
    plot!(test_t ./ (3600 * 24 * 365), plot_test_output[i+1, :], title=species, label="Interpolated", legend=:outertopright)
    xticks!([10^0, 10,10^2,10^3,10^4,10^5,10^6])
    xaxis!(:log10)
    xlims!((1., 10^7))
    ylims!((10^-30, 1))
    yaxis!(:log10)
    savefig("./output/$species.png")
end
