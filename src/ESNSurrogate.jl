
module EchoStateNetworkSurrogate

    using Surrogates, ReservoirComputing, BSON

    mutable struct ESNSurrogate
        time_esn
        species_esn
        time_interpolation
        species_interpolation
    end

    struct PhysicalParameters{T<:Real}
        zeta::T
        omega::T
        T::T
        F_UV::T
        A_v::T
        E::T
        density::T
    end


    function predict(model::ESNSurrogate, p::PhysicalParameters, u0::Vector{Real})
        time_weights = model.time_interpolation(p...)
        species_weights = model.species_interpolation(p...)
        model.time_esn
    end

    predict(model::ESNSurrogate, zeta, omega, T, F_UV, A_v, E, density, u0) = predict(model, PhysicalParameters(zeta, omega, T, F_UV, A_v, E, density), u0)


    function load_model(filepath::String)
        
    end

    function save_model(model::ESNSurrogate, filepath::String)
        
    end


end
