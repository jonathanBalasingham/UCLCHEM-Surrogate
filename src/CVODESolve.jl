using Sundials
using DifferentialEquations
using DelimitedFiles
using DiffEqCallbacks
import DifferentialEquations: solve

"""
Solver a ChemicalNetworkProblem by breaking it into chunks.
Time span of each problem is controlled by `time_factor` 
and `time_factor_post_1000_years`. 

Adaptively solves the problem and returns a ChemicalNetworkSolution
"""
function solve(prob::ChemicalNetworkProblem; 
                abstol::Float64=10^-20, 
                reltol=10^-6, 
                maxiter::Int=10000, 
                solver=CVODE_BDF,
                time_factor=1.1, time_factor_pre_1000_years=10.)
    current_time = 0.0
    target_time = 1.1
    u = Float64[]
    t = Float64[]
    year_1000 = 1000. * 3600. * 24. * 365.
    current_problem = ODEProblem{true}(prob.network, prob.u0, (current_time, target_time), prob.rates)


    while current_time <= prob.tspan[2]
        sol = DifferentialEquations.solve(current_problem, solver(), maxiter=maxiter, reltol=reltol, abstol=abstol, cb=CallbackSet(PositiveDomain(abstol=1e-200)))
        u = vcat(u, sol.u[2:end])
        t = vcat(t, sol.t[2:end])
        current_time = target_time
        if current_time > year_1000
             target_time *= time_factor 
        else 
            target_time *= time_factor_pre_1000_years 
        end
        @async update!(p, current_time |> floor |> Integer)
        current_problem = remake(current_problem, tspan=(current_time, target_time), u0=sol.u[end])
    end
    ChemicalNetworkSolution(t,u, prob.species, prob.rates)
end


"""
Solves a ChemicalNetworkProblem at pre-determined
time steps dictated by `saveat`

Returns a ChemicalNetworkSolution
"""
function solve(prob::ChemicalNetworkProblem,
                saveat::Array; 
                abstol::Float64=10^-30, 
                reltol=10^-8, 
                maxiter::Int=10000, 
                solver=CVODE_BDF,
                time_factor=1.1, time_factor_pre_1000_years=10.)
    current_time = 0.0
    target_time = 1.1
    u = Float64[]
    t = Float64[]
    year_1000 = 1000. * 3600. * 24. * 365.
    current_problem = ODEProblem{true}(prob.network, prob.u0, (current_time, target_time), prob.rates)

    sub_saveat = filter(x -> x < target_time && x > current_time, saveat)
    while current_time <= prob.tspan[2]
        sol = DifferentialEquations.solve(current_problem, 
                                          solver(),
                                          saveat=sub_saveat,
                                          maxiter=maxiter,
                                          reltol=reltol, 
                                          abstol=abstol, 
                                          cb=CallbackSet(PositiveDomain(abstol=1e-200)))
        u = vcat(u, sol.u[2:end])
        t = vcat(t, sol.t[2:end])
        current_time = target_time
        if current_time > year_1000
            target_time *= time_factor 
        else 
            target_time *= time_factor_pre_1000_years 
        end
        @async update!(p, current_time |> floor |> Integer)
        current_problem = remake(current_problem, tspan=(current_time, target_time), u0=sol.u[end])
        sub_saveat = filter(x -> x < target_time && x > current_time, saveat)
    end
    ChemicalNetworkSolution(t,u, prob.species, prob.rates)
end


"""
Solves a ChemicalNetworkProblem at adaptive time
points and writes solution to a CSV file.

nothing returned
"""
function solve(prob::ChemicalNetworkProblem,
               filepath::AbstractString; 
               abstol::Float64=10^-20, 
               reltol=10^-4, 
               maxiter::Int=10000,
               saveat=60, 
               solver=CVODE_BDF)

    current_time = 0.0
    target_time = 1.1
    year_1000 = 1000. * 3600. * 24. * 365.
    time_factor_pre_1000_years = 10.
    time_factor_post_1000_years = 1.1
    current_problem = ODEProblem{true}(prob.network, prob.u0, (current_time, target_time), prob.rates)
    year_in_secs = 3600 * 24 * 365

    open(filepath, "a") do io
        writedlm(io, Array([0.0; prob.u0]'), ',')
    end

    while current_time <= prob.tspan[2]
        println(current_time)
        

        if current_time > year_in_secs
            sol = DifferentialEquations.solve(current_problem, solver(), maxiter=maxiter, reltol=reltol, abstol=abstol, saveat=year_in_secs, dt=1.)
            data = Matrix(hcat(sol.u...))

            open(filepath, "a") do io
                writedlm(io, [sol.t data'], ',')
            end
        else
            sol = DifferentialEquations.solve(current_problem, solver(), maxiter=maxiter, reltol=reltol, abstol=abstol, dt=1.)
            println(length(sol))
        end

        current_time = target_time
        if current_time > year_1000
            target_time *= time_factor_post_1000_years 
        else 
            target_time *= time_factor_pre_1000_years 
        end
        current_problem = remake(current_problem, tspan=(current_time, target_time), u0=sol.u[end])
    end

end


function solve(prob::ChemicalNetworkProblem, saveat::Real; 
    abstol::Float64=10^-20, 
    reltol=10^-6, 
    maxiter::Int=10000, 
    solver=CVODE_BDF,
    time_factor=1.1, time_factor_pre_1000_years=10.)
    current_time = 0.0
    target_time = 1.1
    u = Float64[]
    t = Float64[]
    year_1000 = 1000. * 3600. * 24. * 365.
    current_problem = ODEProblem{true}(prob.network, prob.u0, (current_time, target_time), prob.rates)

    while current_time <= prob.tspan[2]
        sol = DifferentialEquations.solve(current_problem, solver(),saveat=saveat, maxiter=maxiter, reltol=reltol, abstol=abstol, cb=CallbackSet(PositiveDomain(abstol=1e-200)))
        u = vcat(u, sol.u[2:end])
        t = vcat(t, sol.t[2:end])
        current_time = target_time
        if current_time > year_1000
            target_time *= time_factor 
        else 
            target_time *= time_factor_pre_1000_years 
        end
        @async update!(p, current_time |> floor |> Integer)
        current_problem = remake(current_problem, tspan=(current_time, target_time), u0=sol.u[end])
    end
    ChemicalNetworkSolution(t,u, prob.species, prob.rates)
    end