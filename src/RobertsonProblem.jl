using DifferentialEquations, Sundials


function rober(du,u,p,t)
    y₁,y₂,y₃ = u
    k₁,k₂,k₃ = p
    du[1] = -k₁*y₁+k₃*y₂*y₃
    du[2] =  k₁*y₁-k₂*y₂^2-k₃*y₂*y₃
    du[3] =  k₂*y₂^2
    nothing
end
  

function formulate_robertson(u0, rates; tspan=(0., 1e5), saveat=nothing, solver=CVODE_BDF)
    prob = ODEProblem(rober, u0, tspan, rates)
    if !isnothing(saveat)
        sol = solve(prob, saveat=saveat)
    else
        sol = solve(prob)
    end
    train = hcat(sol.u...)
    rates_t = repeat(rates, length(sol.t)) |> x->reshape(x, length(rates), :)

    X = train[:, 1:end-1]
    y = train[:, 2:end]
    X, y
end