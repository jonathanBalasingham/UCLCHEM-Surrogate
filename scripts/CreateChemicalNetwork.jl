#=

This is a very simple example of how to create a chemical 
network for a set of parameters, solve then visualize the
result. For more options, please see GasPhaseNetwork.jl in
the src directory.

=#

using DrWatson
@quickactivate "UCLCHEM Surrogate"

using DifferentialEquations, Plots, Sundials

include.(srcdir.(["GasPhaseNetwork.jl", "CVODESolve.jl", "Visualize.jl"])) 

# This is the reactions filepath and initial conditions filepath
rfp, icfp = map(x -> datadir("exp_raw", x), ["reactions_final.csv", "initcond0.csv"])

#  zeta, omega, T, F_UV, A_v, E, density
example_parameters = [1e-17, 0.5, 10, 1., 10., 1e2]

# Set time span to 10 million years
timespan = (0., 10^7 * 365. * 24. * 3600.)

# Convert the array to Chemical Network Parameters
params = Parameters(example_parameters...)

# formulate then solve
cnp = formulate_all(rfp, icfp, params, tspan=timespan);
sol = solve(cnp)

#plot the desired species and display
visualize(sol, species=["H", "C", "O"])