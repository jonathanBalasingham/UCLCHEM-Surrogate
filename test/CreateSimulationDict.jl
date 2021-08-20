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

dark_cloud_upper = get_rates(rfp, Parameters(rates_set_upper_bound...))
dark_cloud_lower = get_rates(rfp, Parameters(rates_set_lower_bound...))
true_dark_cloud_lower = [min(a,b) for (a,b) in zip(dark_cloud_lower, dark_cloud_upper)]
true_dark_cloud_upper = [max(a,b) for (a,b) in zip(dark_cloud_lower, dark_cloud_upper)]

parameter_samples = sample(10, true_dark_cloud_lower .* 1.05, true_dark_cloud_upper .* .995, SobolSample());

function condition(u, t, integrator)
    check_error(integrator) != :Success
end

function affect!(integrator)
    terminate!(integrator)
end


callback = ContinuousCallback(condition, affect!)

timepoints = deserialize("./timepoints")
bottom = filter(x->x>0,true_dark_cloud_lower) |> minimum |> log10 |> abs |> x->round(x)+1 
r(x) = replace(log10.(x) .+ bottom, -Inf=>0.0)

if isfile("./simulation_dict")
    simulation_dict = deserialize("./simulation_dict")
else
    simulation_dict = Dict()
end

parameter_samples[1:end] .|>
            begin
                x->formulate_all(rfp, icfp, Parameters(zeros(6)...), tspan=tspan, rates=[x...]) |>
                x->
                begin 
                    rates = r(x.rates)
                    try
                        if rates in keys(simulation_dict)
                            @info "skipping"
                            return
                        end
                        prob=ODEProblem(x.network, x.u0, x.tspan)
                        @time sol = solve(prob, CVODE_BDF(), abstol=10e-30, reltol=10e-15, callback=callback, saveat=timepoints)
                        if sol.t[end] >= tspan[2]*.999
                            train = hcat(sol.u...) |> x -> log2.(x .+ abs(minimum(x))*1.01)
                            simulation_dict[rates] = train
                        end
                    catch e
                        println(e)
                    end
                end
            end;

serialize("./simulation_dict", simulation_dict)


if exists("./simulation_dict")