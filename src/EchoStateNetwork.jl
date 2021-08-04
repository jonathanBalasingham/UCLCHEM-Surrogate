module ESN

using Base: AbstractFloat, Integer
using SparseArrays, LinearAlgebra
import Base.:*
using Flux

abstract type AbstractEchoStateNetwork{T} end
abstract type AbstractReservoir{T} end

mutable struct EchoStateReservoir{T<:AbstractFloat} <: AbstractReservoir
    """
    These fields allow this struct to be treated as a 
    recurrent layer in Flux. We do not apply @functor
    as the weights are not meant to be trained. 

    Parameteric Type is added so we consistently use
    32-bit or 64-bit floating point throughtout for 
    performance reasons.
    """
    weight::Matrix{T}
    b::Vector{T}
    f::Function
    state::Vector{T}
end

mutable struct SimpleCyclicReservoir{T<:AbstractFloat} <: AbstractReservoir
    weight::Matrix{T}
    b::Vector{T}
    f::Function
    state::Vector{T}
    feedback::Bool
end

mutable struct DelayLineReservoir{T<:AbstractFloat} <: AbstractReservoir
    weight::Matrix{T}
    b::Vector{T}
    f::Function
    state::Vector{T}
end

EchoStateReservoir(W::Matrix{T}, b::Vector{T}, f::Function) where T<:AbstractFloat = EchoStateReservoir{T}(W,b,f,zeros(T, size(W, 1)))

function EchoStateReservoir{T}(reservoir_size::Int, spectral_radius::A, sparsity::A; activation=identity) where {A<:Real, T<:AbstractFloat}
    reservoir_size < 1 && @error "Invalid reservoir size: $reservoir_size"
    0 < sparsity < 1 || @error "Sparsity out of range of (0,1)"
    
    r = sprand(reservoir_size, reservoir_size, sparsity) |> Matrix{T}
    EchoStateReservoir(convert.(T, r .*  spectral_radius / maximum(abs.(eigvals(r)))), zeros(T, reservoir_size), activation)
end

(res::EchoStateReservoir{T})(input::Vector{T}, α=.1) where T<:AbstractFloat = 
    res.state = (1-convert(T, α)).*res.state + convert(T,α)*res.f.(input + res.weight*res.state)
*(res::EchoStateReservoir{T}, input::Vector{T}, α=.1) where T<:AbstractFloat = res(input, α)

(res::EchoStateReservoir{T})(input::Vector{V}, α=.1) where {T<:AbstractFloat, V<:AbstractFloat} = 
    res.state = (1-convert(T, α)).*res.state + convert(T,α)*res.f.(convert(Vector{T}, input) + res.weight*res.state)
*(res::EchoStateReservoir{T}, input::Vector{V}, α=.1) where {T<:AbstractFloat, V<:AbstractFloat} = res(input, α)

reset!(res::EchoStateReservoir{T}) where T<:AbstractFloat = res.state .*= 0

function Base.show(io::IO, res::EchoStateReservoir{T}) where T<:AbstractFloat
    sparsity  = count(x->x!=0.0, res.weight) / length(res.weight)
    radius = maximum(abs.(eigvals(res.weight)))
    print(io, "EchoStateReservoir{$T}(sparsity: $(sparsity*100)%, radius: $radius, size: $(size(res.weight, 1)))")
end

#=
This could be changed to a Flux Chain, but I need to figure out
a way to make the reservoir to behave recurrently instead of as 
just a normal feed-forward layer.
=#
mutable struct EchoStateNetwork{T<:AbstractFloat} <: AbstractEchoStateNetwork{T}
    input_layer::Dense
    reservoir::EchoStateReservoir{T}
    output_layer::Dense
    # TODO: I should change this to be uniform with the deep esn constructor
    function EchoStateNetwork{T}(input_layer::Dense, W::EchoStateReservoir{T}, output_layer::Dense) where T<:AbstractFloat
        let input_size = size(input_layer.weight, 1), res_size = size(W.weight, 1)
            viable_res_size = Int(floor(res_size/input_size)*input_size)
            input_size == res_size || @error "Incompatible input and reservoir sizes:($input_size, $res_size)
                                                         Reservoir needs size: $viable_res_size"
        end
        new(input_layer, W, output_layer)
    end
end

reset!(esn::EchoStateNetwork{T}) where T<:AbstractFloat = reset!(esn.reservoir)

function EchoStateNetwork{T}(input_size::I, 
                             reservoir_size::I, 
                             output_size::I, 
                             σ=0.5, ρ=1.0; 
                             sparsity=0.3, 
                             input_activation=tanh, 
                             reservoir_activation=tanh, 
                             output_activation=identity) where {I<:Integer, T<:AbstractFloat}
    f = T == Float32 ? f32 : f64
    inp = Dense(input_size, reservoir_size, input_activation, init=Flux.sparse_init(sparsity=sparsity))
    inp.weight .*= σ
    res = EchoStateReservoir{T}(reservoir_size, ρ, sparsity, activation=reservoir_activation)
    out = f(Dense(reservoir_size, output_size, output_activation))
    EchoStateNetwork{T}(f(inp), res, out)
end

mutable struct DeepEchoStateNetwork{T<:AbstractFloat} <: AbstractEchoStateNetwork{T}
    input_layers::Vector{Dense}
    reservoirs::Vector{EchoStateReservoir{T}}
    output_layer::Dense
    function DeepEchoStateNetwork(input_layer::Vector{Dense}, W::Vector{EchoStateReservoir{T}}, output_layer::Dense) where {T<:AbstractFloat}
        let input_size = size(input_layer[begin].weight, 1), res_size = size(W[begin].weight, 1)
            viable_res_size = Int(floor(res_size/input_size)*input_size)
            input_size == res_size || @error "Incompatible input and reservoir sizes:($input_size, $res_size)
                                                         Reservoir needs size: $viable_res_size"
        end
        new{T}(input_layer, W, output_layer)
    end
end

function DeepEchoStateNetwork{T}(input_size::I,
                                reservoir_size::I,
                                layers::I,
                                output_size::I,
                                σ=0.5, ρ=1.0; 
                                sparsity=0.3, 
                                input_activation=tanh, 
                                reservoir_activation=tanh, 
                                output_activation=identity) where {I<:Integer, T<:AbstractFloat}
    f = T == Float32 ? f32 : f64
    inp = Dense[f(Dense(i == 1 ? input_size : reservoir_size, reservoir_size, input_activation, init=Flux.sparse_init(sparsity=sparsity))) for i in 1:layers]
    for i in inp i.weight .*= σ end
    res = EchoStateReservoir{T}[EchoStateReservoir{T}(reservoir_size, ρ, sparsity, activation=reservoir_activation) for i in 1:layers]
    out = f(Dense(reservoir_size*layers, output_size, output_activation))
    DeepEchoStateNetwork(inp, res, out)
end

reset!(desn::DeepEchoStateNetwork) = for res in desn.reservoirs reset!(res) end


(esn::EchoStateNetwork)(x::AbstractArray) = x |> esn.input_layer |> esn.reservoir |> esn.output_layer

stateof(esn::EchoStateNetwork) = esn.reservoir.state
inputdim(esn::EchoStateNetwork) = size(esn.input_layer.weight, 2)
inputdim(desn::DeepEchoStateNetwork) = size(desn.input_layers[begin].weight, 2)
outputdim(esn::T) where T<:AbstractEchoStateNetwork = size(esn.output_layer.weight, 1)


function get_states!(esn::EchoStateNetwork, train::Matrix{T}) where T<:AbstractFloat
    reservoir_output = [x |> esn.input_layer |> esn.reservoir for x in eachcol(train)] 
    ESN.reset!(esn)
    hcat(reservoir_output...)' |> Matrix
end

function get_states!(desn::DeepEchoStateNetwork, train::Matrix{T}) where T<:AbstractFloat
    reservoir_output = T[]
    for input in eachcol(train)
        states = T[]
        for (inp_lay, res) in zip(desn.input_layers, desn.reservoirs)
            input = input |> inp_lay |> res
            append!(states, input)
        end
        if size(reservoir_output) == (0,) reservoir_output = states else reservoir_output = hcat(reservoir_output, states) end
    end
    ESN.reset!(desn)
    reservoir_output' |> Matrix{T}
end


function train!(esn::AbstractEchoStateNetwork{T}, X::Array{T, 3}, y::Array{T, 3}, β=0.01) where T<:AbstractFloat
    let x_size = size(X), y_size = size(y)
        x_size[2] == y_size[2] || @error "X and y do not have the same size of series: $((x_size[2], y_size[2]))"
        x_size[3] == y_size[3] || @error "X and y do not have the same number of sets: $((x_size[3], y_size[3]))"
        inputdim(esn) == x_size[1] || @error "X has the incorrect input dimension, has $(x_size[1]), needs $(inputdim(esn))"
        outputdim(esn) == y_size[1] || @error "y has the incorrect output dimension, has $(y_size[1]), needs $(outputdim(esn))"
    end
    reset!(esn)
    res_output = hcat([get_states!(esn, X[:,:,i]) for i in 1:size(X,3)]'...)
    _y = reshape(y, size(y, 1), :)
    term2 = (res_output*res_output') 
    for i in 1:size(term2, 2) term2[i,i] += β end
    esn.output_layer.weight .= (_y*res_output') * inv(term2)
end

function train!(esn::AbstractEchoStateNetwork{T}, X::Vector{Matrix{T}}, y::Vector{Matrix{T}}, β=0.01) where T<:AbstractFloat
    length(X) == length(y) || @error "X and y do not have the same size of series: $((length(X), length(y)))"
    for i in 1:length(X)
        x_size = size(X[i])
        y_size = size(y[i])
        x_size[2] == y_size[2] || @error "X and y do not have the same size of series: $((x_size[2], y_size[2]))"
        inputdim(esn) == x_size[1] || @error "X has the incorrect input dimension, has $(x_size[1]), needs $(inputdim(esn))"
        outputdim(esn) == y_size[1] || @error "y has the incorrect output dimension, has $(y_size[1]), needs $(outputdim(esn))"
    end
    reset!(esn)
    gs = x->get_states!(esn, x)
    res_output = hcat(gs.(X)'...)
    _y = hcat(y...)
    term2 = (res_output*res_output') 
    for i in 1:size(term2, 2) term2[i,i] += β end
    esn.output_layer.weight .= (_y*res_output') * inv(term2)
end


function predict!(esn::AbstractEchoStateNetwork{T}, input::Matrix{T}; clear_state=true) where {T<:AbstractFloat}
    if clear_state ESN.reset!(esn) end
    hcat([esn(d) for d in eachcol(input)]...)
end

function predict!(esn::EchoStateNetwork{T}, xt::Matrix{T}, st::Matrix{T}; clear_state=true) where T<:AbstractFloat
    size(xt, 1) + size(st, 1) == inputdim(esn) || @error "Dimension of X(t) plus S(t) must be equal to the input dimension of the ESN"
    size(xt,2) >= size(st, 2) && begin @warn "length of X(t) is less than or equal to length of S(t), no prediction will be done"; return end
    warmup_size = size(xt, 2)
    if clear_state ESN.reset!(esn) end

    input = vcat(st[:, 1:warmup_size], xt)
    for d in eachcol(input[:, 1:end-1]) esn(d) end
    pred = esn(input[:, end])
    hcat(pred, [pred = esn(vcat(st[:, i], pred)) for i in size(xt,2)+2:size(st, 2)]...)
end


function (desn::DeepEchoStateNetwork)(input::AbstractArray)
    """
    Deep echo state networks obtain their output by
    passing input through each input layer and reservoir
    consecutively and passing their respective state to
    the output layer in the form of a single concatenated
    vector.
    """
    states = []
    for (inp_lay, res) in zip(desn.input_layers, desn.reservoirs)
        input = input |> inp_lay |> res
        append!(states, input)
    end
    states |> desn.output_layer
end

end

