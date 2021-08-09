include(srcdir("EchoStateNetwork.jl"))
using DataStructures, Surrogates

abstract type Model end

mutable struct MasterNetworkModel{T} <: Model where T<:AbstractReservoir
    esn::EchoStateNetwork
end

mutable struct HybridModel <: Model
    hesn::HybridEchoStateNetwork
end

mutable struct InterpolatedModel <: Model
    esn::EchoStateNetwork
    output_weights::Vector{Matrix}
    parameters::Vector{Vector{Float64}}
    interp
end



mutable struct Model{T} where T <: AbstractEchoStateNetwork
    esn::T
    current::Integer
    parameter_sets
    function Model(esn::T, ub, lb, samples) where T<:AbstractEchoStateNetwork
        new{T}(esn, 1, GridSample())
    end
end

function tune!(m::Model)
    m.current += 1
    remake!(m.esn, m.parameter_sets[m.current]...)
end