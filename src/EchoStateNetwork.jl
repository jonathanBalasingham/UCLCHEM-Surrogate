module ESN

using SparseArrays, LinearAlgebra
import Base.:*
using Flux

mutable struct EchoStateReservoir{T <: AbstractFloat}
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

(res::EchoStateReservoir{T})(input::Vector{T}, α=1.0) where T<:AbstractFloat = 
    res.state = (1-convert(T, α)).*res.state + convert(T,α)*res.f.(input + res.weight*res.state)
*(res::EchoStateReservoir{T}, input::Vector{T}, α=1.0) where T<:AbstractFloat = res(input, α)

(res::EchoStateReservoir{T})(input::Vector{V}, α=1.0) where {T<:AbstractFloat, V<:AbstractFloat} = 
    res.state = (1-convert(T, α)).*res.state + convert(T,α)*res.f.(convert(Vector{T}, input) + res.weight*res.state)
*(res::EchoStateReservoir{T}, input::Vector{V}, α=1.0) where {T<:AbstractFloat, V<:AbstractFloat} = res(input, α)

reset!(res::EchoStateReservoir{T}) where T<:AbstractFloat = res.state .*= 0

#=
This could be changed to a Flux Chain, but I need to figure out
a way to make the reservoir to behave recurrently instead of as 
just a normal feed-forward layer.
=#
mutable struct EchoStateNetwork{T<:AbstractFloat}
    input_layer::Dense
    reservoir::EchoStateReservoir{T}
    output_layer::Dense
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
                             σ=0.5, λ=1.0; 
                             sparsity=0.3, 
                             input_activation=tanh, 
                             reservoir_activation=tanh, 
                             output_activation=identity) where {I<:Integer, T<:AbstractFloat}
    f = T == Float32 ? f32 : f64
    inp = Dense(input_size, reservoir_size, input_activation)
    inp.weight .*= σ
    res = EchoStateReservoir{T}(reservoir_size, λ, sparsity, activation=reservoir_activation)
    out = f(Dense(reservoir_size, output_size, output_activation))
    EchoStateNetwork{T}(f(inp), res, out)
end

(esn::EchoStateNetwork)(x::AbstractArray) = x |> esn.input_layer |> esn.reservoir |> esn.output_layer

stateof(esn::EchoStateNetwork) = esn.reservoir.state
inputdim(esn::EchoStateNetwork) = size(esn.input_layer.weight, 2)
outputdim(esn::EchoStateNetwork) = size(esn.output_layer.weight, 1)

function train!(esn::EchoStateNetwork{T}, X::Array{T, 3}, y::Array{T, 3}, β=0.01) where T<:AbstractFloat
    let x_size = size(X), y_size = size(y)
        x_size[2] == y_size[2] || @error "X and y do not have the same size of series: $((x_size[2], y_size[2]))"
        x_size[3] == y_size[3] || @error "X and y do not have the same number of sets: $((x_size[3], y_size[3]))"
        inputdim(esn) == x_size[1] || @error "X has the incorrect input dimension, has $(x_size[1]), needs $(inputdim(esn))"
        outputdim(esn) == y_size[1] || @error "y has the incorrect output dimension, has $(y_size[1]), needs $(outputdim(esn))"
    end

    function _get_states!(train::Array{Float64, 2})
        reservoir_output = [x |> esn.input_layer |> esn.reservoir for x in eachcol(train)] 
        ESN.reset!(esn)
        hcat(reservoir_output...)' |> Matrix
    end

    res_output = hcat([_get_states!(X[:,:,i]) for i in 1:size(X,3)]'...)
    _y = reshape(y, size(y, 1), :)
    term2 = (res_output*res_output') # dimension is (res size, time series length) -> How to incorporate more series?????
    for i in 1:size(term2, 2) term2[i,i] += β end
    esn.output_layer.weight .= (_y*res_output') * inv(term2)
end

function train!(esn::EchoStateNetwork{T}, X::Vector{Matrix{T}}, y::Vector{Matrix{T}}, β=0.01) where T<:AbstractFloat
    length(X) == length(y) || @error "X and y do not have the same size of series: $((length(X), length(y)))"
    for i in 1:length(X)
        x_size = size(X[i])
        y_size = size(y[i])
        x_size[2] == y_size[2] || @error "X and y do not have the same size of series: $((x_size[2], y_size[2]))"
        inputdim(esn) == x_size[1] || @error "X has the incorrect input dimension, has $(x_size[1]), needs $(inputdim(esn))"
        outputdim(esn) == y_size[1] || @error "y has the incorrect output dimension, has $(y_size[1]), needs $(outputdim(esn))"
    end

    function _get_states!(train::Array{Float64, 2})
        reservoir_output = [x |> esn.input_layer |> esn.reservoir for x in eachcol(train)] 
        ESN.reset!(esn)
        hcat(reservoir_output...)' |> Matrix
    end

    res_output = hcat(_get_states!.(X)'...)
    _y = hcat(y...)
    term2 = (res_output*res_output') 
    for i in 1:size(term2, 2) term2[i,i] += β end
    esn.output_layer.weight .= (_y*res_output') * inv(term2)
end


function predict!(esn::EchoStateNetwork{T}, input::Matrix{T}; clear_state=true) where T<:AbstractFloat
    if clear_state ESN.reset!(esn) end
    hcat([esn(d) for d in eachcol(input)]...)
end

function predict!(esn::EchoStateNetwork{T}, xt::Matrix{T}, st::Matrix{T}) where T<:AbstractFloat
    size(xt, 1) + size(st, 1) == inputdim(esn) || @error "Dimension of X(t) plus S(t) must be equal to the input dimension of the ESN"
    size(xt,2) >= size(st, 2) && begin @warn "length of X(t) is less than or equal to length of S(t), no prediction will be done"; return end
    warmup_size = size(xt, 2)

    input = vcat(xt, st[:, 1:warmup_size])
    for d in eachcol(input[:, 1:end-1]) esn(d) end
    pred = esn(input[:, end])
    hcat(pred, [pred = esn(vcat(st[:, i], pred)) for i in size(xt,2)+2:size(st, 2)]...)
end


end

