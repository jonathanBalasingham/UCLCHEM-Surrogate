
# UCLCHEM Implementation
include.(["GasPhaseNetwork.jl", "CVODESolve.jl", "Visualize.jl"])

# ESN Implementation
import Main.ESN: EchoStateNetwork, DeepEchoStateNetwork

# Surrogate Implementation
include("ESNSurrogate.jl")

include("Transform.jl")


function prepare_warmup(solution; steps=10)
    transform(solution[2:end, begin:steps])
end

function test!(esn, beta, X, y; nrm=false)
    if nrm
        u0 = X[begin][:, begin]
        X_t = transform.(X, u0=u0)
        y_t = transform.(y, u0=u0)
        #ESN.train!(esn, X, y, beta)
        ESN.train!(esn, X_t, y_t, beta)
        warmup_length = 10
        warmup = X_t[begin][:, begin:warmup_length]
        steps = size(y[begin], 2) - size(warmup, 2)
        pred = ESN.predict!(esn, warmup, steps) # |> x->vcat(x, hcat(sum.(eachcol(2 .^ x[1:end, :]))...))
        pred = transform_back(pred, u0=u0)
        _y = y[begin] #|> x->vcat(x, hcat(sum.(eachcol(2 .^ x[1:end, :]))...))
        Flux.Losses.mae(pred, _y[begin:end, warmup_length:end])
    else
        ESN.train!(esn, X, y, beta)
        warmup_length = 10
        warmup = X[begin][:, begin:warmup_length]
        steps = size(y[begin], 2) - size(warmup, 2)
        @time pred = ESN.predict!(esn, warmup, steps) # |> x->vcat(x, hcat(sum.(eachcol(2 .^ x[1:end, :]))...))
        _y = y[begin] #|> x->vcat(x, hcat(sum.(eachcol(2 .^ x[1:end, :]))...))
        Flux.Losses.mae(pred, _y[begin:end, warmup_length:end])
    end
  end
  
function test_all(esn, X, y, beta=5.0, reduction_factor=.5; nrm=false)
    error = Inf
    while true
        new_error = test!(esn, beta, X, y, nrm=nrm)
        if new_error > error || isinf(new_error)
            return (error, beta / reduction_factor)
        else
            error = new_error
        end
        beta *= reduction_factor
    end
end

function train_and_predict(esn, X, y, warmup)
    err, beta = test_all(esn, [X], [y], 1., nrm=true)
    @info "Final regularization parameter of $beta, with MAE of $err"
    ESN.train!(esn, [transform(X)], [transform(y)], beta)
    ESN.predict!(esn, warmup, size(y, 2) - size(warmup, 2) + 1)
end
  

function create_heatmap(Ts, densities, zeta, surrogate; title = "")
    function f(zeta, T, density)
        pa = Parameters(zeta, .5, T, 1., 10., density)
        p = formulate_all(rfp, icfp, pa, tspan=tspan)
        @time sol = solve(ODEProblem(p.network, p.u0, p.tspan), CVODE_BDF(), abstol=1e-25, reltol=1e-7, saveat=surrogate.timepoints)
        train_subset = vcat(sol.t', hcat(sol.u...)) .|> x -> log2.(x .+ abs(minimum(x))*1.01)
    
        X = [train_subset[2:end, begin:end-1]]
        y = [train_subset[2:end, begin+1:end]]
        u0 = get_u0(X[begin])

        interp_rates = r(p.rates)
        warmup_length = 10
        warmup = X[begin][:, begin:warmup_length]

        prediction = EchoStateNetworkSurrogate.predict(surrogate, interp_rates, transform(warmup, u0=u0))|> x->transform_back(x, u0=u0)    
        mae_error = Flux.Losses.mae(y[begin][:, warmup_length:end], prediction[:, begin+1:end])
        roc_error = roc(y[begin][:, warmup_length:end], prediction[:, begin+1:end])
        pe_error = physical_error(y[begin][:, warmup_length:end], prediction[:, begin+1:end])
        (mae_error, roc_error, pe_error)
    end
    
    hm_mae = zeros(length(Ts), length(densities))
    hm_roc = zeros(length(Ts), length(densities))
    hm_pe = zeros(length(Ts), length(densities))

    for t in 1:length(Ts)
        for d in 1:length(densities)
            m,r,p = f(zeta, Ts[t], densities[d])
            hm_mae[t,d] = m
            hm_roc[t,d] = r
            hm_pe[t,d] = p
        end
    end
    
    p = heatmap(densities,Ts, hm_mae, title="MAE", ylabel="T", xlabel="Density")
    p2 = heatmap(densities,Ts, hm_roc, title="ROC Error", ylabel="T", xlabel="Density")
    p3 = heatmap(densities,Ts, hm_pe, title="Physical Error", ylabel="T", xlabel="Density")
    (p, p2, p3)
end



function create_heatmap(Ts, densities, zeta, surrogate, validation)
    function f(zeta, T, density)
        pa = Parameters(zeta, .5, T, 1., 10., density)
        p = formulate_all(rfp, icfp, pa, tspan=tspan)
        train_subset = validation[r(p.rates)]

        X = [train_subset[2:end, begin:end-1]]
        y = [train_subset[2:end, begin+1:end]]
        u0 = get_u0(X[begin])

        interp_rates = r(p.rates)
        warmup_length = 10
        warmup = X[begin][:, begin:warmup_length]

        prediction = EchoStateNetworkSurrogate.predict(surrogate, interp_rates, transform(warmup, u0=u0))|> x->transform_back(x, u0=u0)    
        mae_error = Flux.Losses.mae(y[begin][:, warmup_length:end], prediction[:, begin+1:end])
        roc_error = roc(y[begin][:, warmup_length:end], prediction[:, begin+1:end])
        pe_error = physical_error(y[begin][:, warmup_length:end], prediction[:, begin+1:end])
        (mae_error, roc_error, pe_error)
    end
    
    hm_mae = zeros(length(Ts), length(densities))
    hm_roc = zeros(length(Ts), length(densities))
    hm_pe = zeros(length(Ts), length(densities))

    for t in 1:length(Ts)
        for d in 1:length(densities)
            m,r,p = f(zeta, Ts[t], densities[d])
            hm_mae[t,d] = m
            hm_roc[t,d] = r
            hm_pe[t,d] = p
        end
    end
    
    p = heatmap(densities,Ts, hm_mae, title="MAE", ylabel="T", xlabel="Density")
    p2 = heatmap(densities,Ts, hm_roc, title="ROC Error", ylabel="T", xlabel="Density")
    p3 = heatmap(densities,Ts, hm_pe, title="Physical Error", ylabel="T", xlabel="Density")
    (p, p2, p3)
end