
module EchoStateNetworkSurrogate

    using Surrogates, Serialization
    include("EchoStateNetwork.jl")
    import Main.ESN: AbstractEchoStateNetwork, predict!

    mutable struct ESNSurrogate{F<:AbstractFloat, R<:RadialBasis}
        esn
        interpolation::R
        timepoints::Vector{F}
    end

    function predict(model::ESNSurrogate, rates::Vector{<:Real}, warmup::Matrix{<:Real})
        model.esn.output_layer.weight .= reshape(model.interpolation(rates), size(model.esn.output_layer.weight))
        warmup_length = size(warmup, 2)
        predict!(model.esn, warmup, length(model.timepoints) - warmup_length)
    end

    function load_model(filepath::String)
        deserialize(filepath)
    end

    function save_model(model::ESNSurrogate, filepath::String=pwd())
        serialize(filepath, model)
    end

end
