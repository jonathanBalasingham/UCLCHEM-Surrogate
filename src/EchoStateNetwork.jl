module ESN

using Distributions: length
using SparseArrays, LinearAlgebra
import Base.:*
using Flux
using Flux: @epochs
using Flux.Data

using Distributions
import DiffEqBase: ODEProblem
using DifferentialEquations
using IterTools: ncycle 

abstract type AbstractEchoStateNetwork{T, R} end
abstract type AbstractReservoir{T} end

mutable struct EchoStateReservoir{T<:AbstractFloat} <: AbstractReservoir{T}
    """
    These fields allow this struct to be treated as a 
    recurrent layer in Flux. We do not apply @functor
    as the weights are not meant to be trained. 

    Parameteric Type is added so we consistently use
    32-bit or 64-bit floating point throughtout to 
    keep memory allocations consistent.
    """
    weight::Matrix{T}
    b::Vector{T}
    f::Function
    state::Vector{T}
    α::T
    function EchoStateReservoir{T}(reservoir_size::Int; spectral_radius::A, sparsity::A, activation=identity, α=1.0) where {A<:Real, T<:AbstractFloat}
        reservoir_size < 1 && @error "Invalid reservoir size: $reservoir_size"
        0 < sparsity < 1 || @error "Sparsity out of range of (0,1)"
        
        r = sprand(reservoir_size, reservoir_size, sparsity) |> Matrix{T}
        new{T}(convert.(T, r .*  spectral_radius / maximum(abs.(eigvals(r)))), zeros(T, reservoir_size), activation, zeros(T, reservoir_size), α)
    end
end


mutable struct DelayLineReservoir{T<:AbstractFloat} <: AbstractReservoir{T}
    """
    """
    weight::Matrix{T}
    b::Vector{T}
    f::Function
    state::Vector{T}
    α::T
    function DelayLineReservoir{T}(reservoir_size::Int, c::T, feedback::T=0.0; activation=tanh, α=1.0, bias=false) where T<:AbstractFloat
        W = zeros(T, reservoir_size, reservoir_size)
        for i in 2:reservoir_size W[i, i-1] = c end 
        if feedback != 0 
            for i in 2:reservoir_size W[i-1,i] = feedback end  
        end
        new(W, bias ? rand(T, reservoir_size) : zeros(T, reservoir_size), activation, zeros(T, reservoir_size), α)
    end
end

mutable struct SimpleCycleReservoir{T<:AbstractFloat} <: AbstractReservoir{T}
    """
    """
    weight::Matrix{T}
    b::Vector{T}
    f::Function
    state::Vector{T}
    α::T
    function SimpleCycleReservoir{T}(reservoir_size::Int, c::T; activation=tanh, α=1.0, bias=false) where T<:AbstractFloat
        W = zeros(T, reservoir_size, reservoir_size)
        for i in 2:reservoir_size W[i, i-1] = c end 
        W[begin, end] = c
        new(W, bias ? rand(T, reservoir_size) : zeros(T, reservoir_size), activation, zeros(T, reservoir_size), α)
    end
end

EchoStateReservoir(W::Matrix{T}, b::Vector{T}, f::Function, α=1.0) where T<:AbstractFloat = EchoStateReservoir{T}(W,b,f,zeros(T, size(W, 1)),α)


(res::AbstractReservoir{T})(input::Vector{T}, α=res.α) where T<:AbstractFloat = 
    res.state = (1-convert(T, α)).*res.state + convert(T,α)*res.f.(input + res.weight*res.state)
*(res::AbstractReservoir{T}, input::Vector{T}, α=res.α) where T<:AbstractFloat = res(input, α)

(res::AbstractReservoir{T})(input::Vector{V}, α=res.α) where {T<:AbstractFloat, V<:AbstractFloat} = 
    res.state = (1-convert(T, α)).*res.state + convert(T,α)*res.f.(convert(Vector{T}, input) + res.weight*res.state)
*(res::AbstractReservoir{T}, input::Vector{V}, α=res.α) where {T<:AbstractFloat, V<:AbstractFloat} = res(input, α)

reset!(res::R) where {R<:AbstractReservoir} = res.state .*= 0

function Base.show(io::IO, res::EchoStateReservoir{T}) where T<:AbstractFloat
    sparsity  = count(x->x!=0.0, res.weight) / length(res.weight)
    radius = maximum(abs.(eigvals(res.weight)))
    print(io, "EchoStateReservoir{$T}(sparsity: $(sparsity*100)%, radius: $radius, size: $(size(res.weight, 1)), leakage: $(res.α))")
end


function create_reservoir(res_type::DataType, float::DataType, size::Integer; kwargs...)
    if res_type <: EchoStateReservoir
        EchoStateReservoir{float}(size, sparsity=:sparsity in keys(kwargs) ? kwargs[:sparsity] : .3, 
                                 spectral_radius=:spectral_radius in keys(kwargs) ? kwargs[:spectral_radius] : 1.0,
                                 α=:α in keys(kwargs) ? kwargs[:α] : 1.0)
    elseif res_type <: SimpleCycleReservoir
        SimpleCycleReservoir{float}(size, kwargs[:c], 
                                    activation=:activation in keys(kwargs) ? kwargs[:activation] : tanh,
                                    α=:α in keys(kwargs) ? kwargs[:α] : 1.0)
    elseif res_type <: DelayLineReservoir
        DelayLineReservoir{float}(size, kwargs[:c], :feedback in keys(kwargs) ? kwargs[:feedback] : 0.0,
                                 activation=:activation in keys(kwargs) ? kwargs[:activation] : tanh,
                                 α=:α in keys(kwargs) ? kwargs[:α] : 1.0)
    end
end

#=
This could be changed to a Flux Chain, but I need to figure out
a way to make the reservoir to behave recurrently instead of as 
just a normal feed-forward layer.
=#
mutable struct EchoStateNetwork{T<:AbstractFloat,R<:AbstractReservoir{T}} <: AbstractEchoStateNetwork{T, R}
    input_layer::Dense
    reservoir::R
    output_layer::Union{Dense, Chain}
    # TODO: I should change this to be uniform with the deep esn constructor
    function EchoStateNetwork{T, R}(input_layer::Dense, W::R, output_layer::Union{Dense, Chain}) where {T<:AbstractFloat, R<:AbstractReservoir{T}}
        let input_size = size(input_layer.weight, 1), res_size = size(W.weight, 1)
            viable_res_size = Int(floor(res_size/input_size)*input_size)
            input_size == res_size || @error "Incompatible input and reservoir sizes:($input_size, $res_size)
                                                         Reservoir needs size: $viable_res_size"
        end
        new(input_layer, W, output_layer)
    end
end

reset!(esn::AbstractEchoStateNetwork) = reset!(esn.reservoir)

function EchoStateNetwork{T, R}(input_size::I, 
                             reservoir_size::I, 
                             output_size::I, 
                             σ=0.5; 
                             input_activation=tanh, 
                             input_sparsity=.7,
                             output_activation=identity,
                             kwargs...) where {I<:Integer, T<:AbstractFloat, R<:AbstractReservoir{T}}
    f = T == Float32 ? f32 : f64
    inp = Dense(input_size, reservoir_size, input_activation, init=Flux.sparse_init(sparsity=input_sparsity))
    inp.weight .*= σ
    res = create_reservoir(R, T, reservoir_size; kwargs...)
    out = f(Dense(reservoir_size, output_size, output_activation))
    EchoStateNetwork{T, R}(f(inp), res, out)
end

mutable struct DeepEchoStateNetwork{T<:AbstractFloat, R<:AbstractReservoir{T}} <: AbstractEchoStateNetwork{T, R}
    input_layers::Vector{Dense}
    reservoirs::Vector{R}
    output_layer::Union{Dense, Chain}
    function DeepEchoStateNetwork(input_layer::Vector{Dense}, W::Vector{R}, output_layer::Union{Dense, Chain}) where {T<:AbstractFloat, R<:AbstractReservoir{T}}
        let input_size = size(input_layer[begin].weight, 1), res_size = size(W[begin].weight, 1)
            viable_res_size = Int(floor(res_size/input_size)*input_size)
            input_size == res_size || @error "Incompatible input and reservoir sizes:($input_size, $res_size)
                                                         Reservoir needs size: $viable_res_size"
        end
        new{T, R}(input_layer, W, output_layer)
    end
end

function DeepEchoStateNetwork{T, R}(input_size::I,
                                reservoir_size::I,
                                layers::I,
                                output_size::I,
                                σ=0.5; 
                                input_activation=tanh, 
                                input_sparsity = 0.3,
                                output_activation=identity,
                                kwargs...) where {I<:Integer, T<:AbstractFloat, R<:AbstractReservoir{T}}
    f = T == Float32 ? f32 : f64
    inp = Dense[f(Dense(i == 1 ? input_size : reservoir_size, reservoir_size, input_activation, init=Flux.sparse_init(sparsity=input_sparsity))) for i in 1:layers]
    for i in inp i.weight .*= σ end
    res = R[create_reservoir(R, T, reservoir_size; kwargs...) for i in 1:layers]
    out = f(Dense(reservoir_size*layers, output_size, output_activation))
    DeepEchoStateNetwork(inp, res, out)
end

reset!(desn::DeepEchoStateNetwork) = for res in desn.reservoirs reset!(res) end


(esn::EchoStateNetwork)(x::T) where T<:AbstractArray = x |> esn.input_layer |> esn.reservoir |> esn.output_layer

stateof(esn::EchoStateNetwork) = esn.reservoir.state
inputdim(esn::EchoStateNetwork) = size(esn.input_layer.weight, 2)
inputdim(desn::DeepEchoStateNetwork) = size(desn.input_layers[begin].weight, 2)
outputdim(esn::T) where T<:AbstractEchoStateNetwork = if esn.output_layer isa Dense size(esn.output_layer.weight, 1) else size(esn.output_layer[end].weight, 1) end


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


function train!(esn::AbstractEchoStateNetwork{T, R}, X::Array{T, 3}, y::Array{T, 3}, β=0.01;
                opt=ADAM(0.1), epochs=10, loss=(x,y) -> Flux.Losses.mse(esn.output_layer(x), y)) where {T<:AbstractFloat, R<:AbstractReservoir{T}}
    let x_size = size(X), y_size = size(y)
        x_size[2] == y_size[2] || @error "X and y do not have the same size of series: $((x_size[2], y_size[2]))"
        x_size[3] == y_size[3] || @error "X and y do not have the same number of sets: $((x_size[3], y_size[3]))"
        inputdim(esn) == x_size[1] || @error "X has the incorrect input dimension, has $(x_size[1]), needs $(inputdim(esn))"
        outputdim(esn) == y_size[1] || @error "y has the incorrect output dimension, has $(y_size[1]), needs $(outputdim(esn))"
    end
    reset!(esn)
    res_output = hcat([get_states!(esn, X[:,:,i]) for i in 1:size(X,3)]'...)
    _y = reshape(y, size(y, 1), :)

    if esn.output_layer isa Dense
        term2 = (res_output*res_output') 
        for i in 1:size(term2, 2) term2[i,i] += β end
        esn.output_layer.weight .= (_y*res_output') * inv(term2)
    elseif esn.output_layer isa Chain
        @info "Readout is an MLP, using iterative optimization."
        ps = Flux.params(esn.output_layer)
        data = zip(eachcol(res_output), eachcol(_y)) |> collect
        @info "Starting loss $(loss(data[begin][1], data[begin][2]))"
        @epochs epochs Flux.train!(loss, ps, data, opt)
        @info "Ending loss $(loss(data[begin][1], data[begin][2]))"
    end
end

function train!(esn::AbstractEchoStateNetwork{T, R}, X::Vector{Matrix{T}}, y::Vector{Matrix{T}}, β=0.01; 
                opt=ADAM(0.1), epochs=10, loss=(x,y) -> Flux.Losses.mse(esn.output_layer(x), y), use_gpu=true) where {T<:AbstractFloat, R<:AbstractReservoir{T}}
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

    if esn.output_layer isa Dense
        term2 = (res_output*res_output') 
        for i in 1:size(term2, 2) term2[i,i] += β end
        esn.output_layer.weight .= (_y*res_output') * inv(term2)
    elseif esn.output_layer isa Chain
        @info "Readout is an MLP, using iterative optimization."
        if use_gpu esn.output_layer |> gpu end
        ps = Flux.params(esn.output_layer)
        full_data = zip(eachcol(res_output), eachcol(_y)) |> collect
        total_loss = sum([loss(x, y) for (x,y) in full_data])
        @info "Starting loss $total_loss"
        train_loader = DataLoader(full_data, batchsize=500, shuffle=true)
        for data in train_loader
            @epochs epochs Flux.train!(loss, ps, ncycle(data, 5), opt)
        end
        total_loss = sum([loss(x, y) for (x,y) in full_data])
        @info "Ending loss $total_loss"
    end
end


function predict!(esn::AbstractEchoStateNetwork{T, R}, input::Matrix{T}; clear_state=true) where {T<:AbstractFloat, R<:AbstractReservoir{T}}
    if clear_state ESN.reset!(esn) end
    hcat([esn(d) for d in eachcol(input)]...)
end

function predict!(esn::AbstractEchoStateNetwork{T, R}, warmup::Matrix{T}, steps::Integer; clear_state=true) where {T<:AbstractFloat, R<:AbstractReservoir{T}}
    prediction = ESN.predict!(esn, warmup)[:, end] # warmup the reservoir
    for i in 1:steps
        prediction = hcat(prediction, esn(prediction[:, end]))
    end
    prediction
end

function predict!(esn::AbstractEchoStateNetwork{T, R}, xt::Matrix{T}, st::Matrix{T}; clear_state=true) where {T<:AbstractFloat, R<:AbstractReservoir{T}}
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

mutable struct HybridEchoStateNetwork{T<:AbstractFloat, R<:AbstractReservoir{T}} <: AbstractEchoStateNetwork{T, R}
    input_layer::Dense
    reservoir::R
    output_layer::Union{Dense, Chain}
    prob::ODEProblem
    _integrator
    dt::T
    function HybridEchoStateNetwork{T, R}(input_size::I,
                                        reservoir_size::I,
                                        output_size::I,
                                        problem::ODEProblem,
                                        solver,
                                        dt,
                                        σ = .5;
                                        abstol = 1e-5,
                                        reltol=1e-3,
                                        input_activation=identity, 
                                        output_activation=identity,
                                        kwargs...) where {I<:Integer, T<:AbstractFloat, R<:AbstractReservoir{T}}
        additional_solver_input = length(problem.u0)
        input_layer = Dense(input_size + additional_solver_input, reservoir_size, input_activation)

        rand_indx = rand(1:size(input_layer.weight, 1), 1, reservoir_size)
        rand_weight = rand(Uniform(-σ, σ), input_size)
        new_weights = zeros(T, size(input_layer.weight))

        for (i, (j, k)) in enumerate(zip(rand_indx, rand_weight))
            new_weights[j, i] = k 
        end

        input_layer.weight .= new_weights

        reservoir = create_reservoir(R, T, reservoir_size; kwargs...)
        output_layer = Dense(reservoir_size + additional_solver_input, output_size, output_activation)
        intg = init(problem, solver())
        intg.opts.abstol = abstol
        intg.opts.reltol = reltol
        new{T, R}(input_layer, reservoir, output_layer, problem, intg, dt)
    end
end

inputdim(hesn::HybridEchoStateNetwork) = size(hesn.input_layer.weight, 2) - length(hesn.prob.u0)

function (hesn::HybridEchoStateNetwork)(input::AbstractArray)
    """
    We can't use a single step here because
    a timestep can vary in size. To work around
    this, we need the user to pass a target time 
    starting from t = 0, to simulate to and we 
    pass have the matrix of the solution. We 
    won't know ahead of time the size of this.
    """
    hesn.prob.u0 .= input
    hesn._integrator = init(hesn.prob, hesn._integrator.alg)
    step!(hesn._integrator, hesn.dt)
    temp_solution = hesn._integrator.u
    #append!(hesn.t, hesn.t[end] + hesn._integrator.t)
    vcat(input, temp_solution) |> 
              hesn.input_layer |> 
              hesn.reservoir   |> 
              x-> vcat(x, temp_solution) |> 
              hesn.output_layer    
end


function get_states!(hesn::HybridEchoStateNetwork, train::Matrix{T}) where T<:AbstractFloat
    states = T[]
    for i in 1:size(train, 2)
        hesn.prob.u0 .= train[:, i]
        hesn._integrator = init(hesn.prob, hesn._integrator.alg)
        step!(hesn._integrator)
        temp_solution = hesn._integrator.u
    
        ns = vcat(train[:, i], temp_solution) |> 
                  hesn.input_layer |> 
                  hesn.reservoir   |> 
                  x->vcat(x, temp_solution)
        if isempty(states) states = ns else states = hcat(states, ns) end
    end
    return Matrix(states')
end

function reset!(hesn::HybridEchoStateNetwork)
    reset!(hesn.reservoir)
    abstol = hesn._integrator.opts.abstol
    reltol = hesn._integrator.opts.reltol
    hesn._integrator = init(hesn.prob, hesn._integrator.alg)
    hesn._integrator.opts.abstol = abstol
    hesn._integrator.opts.reltol = reltol
end

mutable struct SplitEchoStateNetwork{T<:AbstractFloat, R<:AbstractReservoir{T}} <: AbstractEchoStateNetwork{T, R}
    input_layers::Vector{Dense}
    reservoirs::Vector{R}
    output_layer::Dense
    function SplitEchoStateNetwork{T, R}(input_sizes::Tuple, 
                                         reservoir_sizes::Tuple, 
                                         output_size,
                                         σ=0.5; 
                                         input_activation=tanh, 
                                         input_sparsity = 0.3,
                                         output_activation=identity,
                                         kwargs...) where {T<:AbstractFloat, R<:AbstractReservoir{T}}
        length(input_sizes) == length(reservoir_sizes) || @error "Input and Reservoir sizes must have equal length"
        f = T == Float32 ? f32 : f64
        inp = Dense[f(Dense(input_sizes[i], reservoir_sizes[i], input_activation, init=Flux.sparse_init(sparsity=input_sparsity))) for i in 1:length(input_sizes)]
        for i in inp i.weight .*= σ end
        res = R[create_reservoir(R, T, reservoir_sizes[i]; kwargs...) for i in 1:length(reservoir_sizes)]
        state_output_size = sum(reservoir_sizes)
        out = f(Dense(state_output_size, output_size, output_activation))
        new{T, R}(inp, res, out)
    end
end

inputdim(sesn::SplitEchoStateNetwork; split=false) = 
    if split 
        [size(d.weight, 2) for d in sesn.input_layers] 
    else 
        sum([size(d.weight, 2) for d in sesn.input_layers]) 
    end

function (sesn::SplitEchoStateNetwork)(input::AbstractArray)
    """

    """
    input_sizes = inputdim(sesn, split=true)    
    end_inds = [0, cumsum(input_sizes)...]
    [input[end_inds[i]+1:end_inds[i+1]] for (i,j) in enumerate(input_sizes)] .|> 
                                                           sesn.input_layers .|>
                                                              sesn.reservoirs |> 
                                                                x->vcat(x...) |> 
                                                              sesn.output_layer
end

function get_states!(sesn::SplitEchoStateNetwork{T}, train::Matrix{T}) where T<:AbstractFloat
    input_sizes = [size(d.weight, 2) for d in sesn.input_layers]
    
    end_inds = [0, cumsum(input_sizes)...]

    states = eachcol(train) .|> input -> begin
        [input[end_inds[i]+1:end_inds[i+1]] for (i,j) in enumerate(input_sizes)] .|> 
                                                            sesn.input_layers .|>
                                                                sesn.reservoirs |> 
                                                                x->vcat(x...)
    end
    hcat(states...)' |> Matrix
end

reset!(sesn::SplitEchoStateNetwork) = for res in sesn.reservoirs reset!(res) end


end
