using Surrogates
include(srcdir("Rates.jl"))
include(srcdir("GasPhaseNetwork.jl"))

function sample_around(n::Integer, rfp::String,physical_parameters::Parameters; method=SobolSample, above::Real=1.1, below::Real=.9)
    rates = get_rates(rfp, physical_parameters)
    Surrogates.sample(n, below .* rates, above .* rates, method()) .|> x->[x...]
end