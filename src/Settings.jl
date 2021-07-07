using ReservoirComputing
include("Rates.jl")


mutable struct ESNSettings
    reservoir_size::Integer
    radius::Float64
    degree::Integer
    activation::Function
    alpha::Float64
    sigma::Float64
    nla_type
    extended_states::Bool
    beta::Float64
    W_path::String
    W_in_path::String
end

mutable struct ChemistrySettings
    zeta::Float64
    omega::Float64
    T::Float64
    F_UV::Float64
    A_v::Float64
    density::Float64
    params::Parameters
    number_of_species::Integer
    number_of_reactions::Integer
    abstol::Float64
    reltol::Float64
    tspan::Tuple
    solver
end

ChemistrySettings() = ChemistrySettings(0.,0.,0.,0.,0.,0.,Parameters(0,0,0,0,0,0),0,0,0,0,(0.,0.),nothing)


struct RolledSimulationSettings
    esn_settings::ESNSettings
    chemistry::ChemistrySettings
end

struct UnrolledSimulationSettings
    reservoir_size::Integer
    radius::Float64
    degree::Integer
    activation::Function
    alpha::Float64
    sigma::Float64
    nla_type
    extended_states::Bool
    beta::Float64
    W_path::String
    W_in_path::String
    zeta::Float64
    omega::Float64
    T::Float64
    F_UV::Float64
    A_v::Float64
    density::Float64
    params::Parameters
    number_of_species::Integer
    number_of_reactions::Integer
    abstol::Float64
    reltol::Float64
    tspan::Tuple{2, Float64}
    solver
end

function roll(uss::UnrolledSimulationSettings)
    esns = ESNSettings(
        uss.reservoir_size,
        uss.radius,
        uss.degree,
        uss.activation,
        uss.alpha,
        uss.sigma,
        uss.nla_type,
        uss.extended_states,
        uss.beta,
        uss.W_path,
        uss.W_in_path
    )
    cs = ChemistrySettings(
        uss.zeta,
        uss.omega,
        uss.T,
        uss.F_UV,
        uss.A_v,
        uss.density,
        uss.params,
        uss.number_of_species,
        uss.number_of_reactions,
        uss.abstol,
        uss.reltol,
        uss.tspan,
        uss.solver
    )
    RolledSimulationSettings(esns, cs)
end


function unroll(rss::RolledSimulationSettings)
    UnrolledSimulationSettings(
        rss.reservoir_size,
        rss.radius,
        rss.degree,
        rss.activation,
        rss.alpha,
        rss.sigma,
        rss.nla_type,
        rss.extended_states,
        rss.beta,
        rss.W_path,
        rss.W_in_path,
        rss.zeta,
        rss.omega,
        rss.T,
        rss.F_UV,
        rss.A_v,
        rss.density,
        rss.params,
        rss.number_of_species,
        rss.number_of_reactions,
        rss.abstol,
        rss.reltol,
        rss.tspan,
        rss.solver
    )
end

toDict(x::T) where {T} = Dict(string(fn)=>getfield(x, fn) for fn ∈ fieldnames(T))

function fromDictToUnrolled(d::Dict)
    UnrolledSimulationSettings(
        s["reservoir_size"],
        s["radius"],
        s["degree"],
        s["activation"],
        s["alpha"],
        s["sigma"],
        s["nla_type"],
        s["extended_states"],
        s["beta"],
        s["W_path"],
        s["W_in_path"],
        s["zeta"],
        s["omega"],
        s["T"],
        s["F_UV"],
        s["A_v"],
        s["density"],
        s["params"],
        s["number_of_species"],
        s["number_of_reactions"],
        s["abstol"],
        s["reltol"],
        s["tspan"],
        s["solver"]
    )
end

fromDictToRolled(d::Dict) = fromDictToUnrolled(d) |> roll