using DrWatson
@quickactivate "UCLCHEM Surrogate"

using DifferentialEquations, Sundials, Serialization, Surrogates

using Random
Random.seed!(0)

include.(srcdir.(["GasPhaseNetwork.jl", "CVODESolve.jl", "Visualize.jl"])) #, "NetworkSurrogate.jl"]))

rfp, icfp, sfp = map(x -> datadir("exp_raw", x), ["reactions_final.csv", "initcond0.csv", "species.csv"])

tspan = (0., 10^7 * 365. * 24. * 3600.)
                      #  zeta, omega, T, F_UV, A_v, E, density
rates_set_lower_bound = [1e-17, 0.5, 10, 1., 10., 1e2]
rates_set_upper_bound = [1e-14, 0.5, 300, 1., 10., 1e6]

dark_cloud_upper = get_rates(rfp, Parameters(rates_set_upper_bound...))
dark_cloud_lower = get_rates(rfp, Parameters(rates_set_lower_bound...))
true_dark_cloud_lower = [min(a,b) for (a,b) in zip(dark_cloud_lower, dark_cloud_upper)]
true_dark_cloud_upper = [max(a,b) for (a,b) in zip(dark_cloud_lower, dark_cloud_upper)]

#parameter_samples = sample(100, true_dark_cloud_lower, true_dark_cloud_upper, SobolSample())
parameter_samples = deserialize("rates_samples");

function condition(u, t, integrator)
    check_error(integrator) != :Success
end

function affect!(integrator)
    terminate!(integrator)
end


callback = ContinuousCallback(condition, affect!)

#timepoints = deserialize("./timepoints")
bottom = filter(x->x>0,true_dark_cloud_lower) |> minimum |> log10 |> abs |> x->round(x)+1 
r(x) = replace(log10.(x) .+ bottom, -Inf=>0.0)

if isfile("./adaptive_simulation_dict")
    adaptive_simulation_dict = deserialize("./adaptive_simulation_dict")
else
    adaptive_simulation_dict = Dict()
end

parameter_samples[98:end] .|>
            begin
                x->formulate_all(rfp, icfp, Parameters(zeros(6)...), tspan=tspan, rates=[x...]) |>
                x->
                begin 
                    rates = r(x.rates)
                    try
                        if rates in keys(adaptive_simulation_dict)
                            @info "skipping"
                            return
                        end
                        prob=ODEProblem(x.network, x.u0, x.tspan)
                        @time sol = solve(prob, CVODE_BDF(), abstol=10e-25, reltol=10e-8, callback=callback) #, saveat=timepoints)
                        if sol.t[end] >= tspan[2]*.999
                            train = vcat(sol.t',hcat(sol.u...) |> x -> log2.(x .+ abs(minimum(x))*1.01))
                            adaptive_simulation_dict[rates] = train
                            serialize("./adaptive_simulation_dict", adaptive_simulation_dict)
                        end
                    catch e
                        println(e)
                    end
                end
            end;


serialize("./adaptive_simulation_dict", adaptive_simulation_dict)

all_timepoints = values(adaptive_simulation_dict) .|> x->x[begin, :]


if isfile("./static_simulation_dict")
    static_simulation_dict = deserialize("./static_simulation_dict")
else
    static_simulation_dict = Dict()
end

timepoints = static_simulation_dict |> first |> x->x.second[begin,:]


parameter_samples[75:end] .|>
            begin
                x->formulate_all(rfp, icfp, Parameters(zeros(6)...), tspan=tspan, rates=[x...]) |>
                x->
                begin 
                    rates = r(x.rates)
                    try
                        if rates in keys(static_simulation_dict)
                            @info "skipping"
                            return
                        end
                        prob=ODEProblem(x.network, x.u0, x.tspan)
                        @time sol = solve(prob, CVODE_BDF(), abstol=10e-25, reltol=10e-8, callback=callback, saveat=timepoints)
                        if sol.t[end] >= tspan[2]*.999
                            train = vcat(sol.t',hcat(sol.u...) |> x -> log2.(x .+ abs(minimum(x))*1.01))
                            static_simulation_dict[rates] = train
                            serialize("./static_simulation_dict", static_simulation_dict)
                        end
                    catch e
                        println(e)
                    end
                end
            end;


serialize("./static_simulation_dict", static_simulation_dict)
