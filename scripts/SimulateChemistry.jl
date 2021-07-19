using DrWatson
using DelimitedFiles, Surrogates
@quickactivate "UCLCHEM Surrogate"
DrWatson.greet()

include.(srcdir.(["GasPhaseNetwork.jl", "CVODESolve.jl", "Visualize.jl"])) #, "NetworkSurrogate.jl"]))

rfp, icfp, sfp = map(x -> datadir("exp_raw", x), ["reactions_final.csv", "initcond4.csv", "species_final.csv"])

tspan = (0., 10^7 * 365. * 24. * 3600.)
                      #  zeta, omega, T, F_UV, A_v, E, density
rates_set_lower_bound = [1e-17, 0.5, 10, 0.5, 2., 1e2]
rates_set_upper_bound = [1., 0.5, 100, 1.5, 10., 1e5]

parameter_samples = sample(1000, rates_set_lower_bound, rates_set_upper_bound, SobolSample())

write_path(x) = datadir("sims", "Adaptive_unstable", x)

for parameter_sample in parameter_samples[1:end]
    pa = Parameters(parameter_sample...)
    p = formulate_all(rfp, icfp, pa)
    try
        @time sol = solve(p, solver=CVODE_BDF)
        train_subset = vcat(sol.t', hcat(sol.u...))
        filename = "CVODE_adaptive" * reduce(*, parameter_sample .|> x -> "_" * string(x)) * ".csv"
        open(write_path(filename), "w") do io
            writedlm(io, train_subset, ',')
        end
    catch e
        @info "errored on $parameter_sample"
        continue
    end
end

write_path(x) = datadir("sims", "Static", x)
saveat = collect(10 .^ (-9:.01:log10(tspan[end])+.1))

for parameter_sample in parameter_samples
    pa = Parameters(parameter_sample...)
    p = formulate_all(rfp, icfp, pa)
    @time sol = solve(p, saveat, solver=CVODE_BDF)
    train_subset = vcat(sol.t', hcat(sol.u...))
    filename = "CVODE_static" * reduce(*, parameter_sample .|> x -> "_" * string(x)) * ".csv"
    open(write_path(filename), "w") do io
        writedlm(io, train_subset, ',')
    end
end

