using Base: Tuple

abstract type Problem end

struct RobertsonProblem <: Problem
    rober::Function
    tspan::Tuple{Float64, Float64}
    u0::Vector{Float64}
    rates
end

function robertson_problem(;kwargs...)
    
end