using Statistics

function mae(truth::T, pred::T) where {T <: Matrix}
    abs.(truth .- pred) |>
        x -> sum(eachrow(x)) |>
        x -> mean(x)
end

function mae(truth::T, pred::T) where {T <: Vector}
    abs.(truth .- pred) |>
        x -> mean(x)
end

function roc(truth::T, pred::T) where {T <: Matrix}
    roc_truth = truth[2:end, :] .- truth[1:end-1, :]
    roc_pred = pred[2:end, :] .- pred[1:end-1, :]
    mae(roc_truth, roc_pred)
end

function roc(truth::T, pred::T) where {T <: Vector}
    roc_truth = truth[2:end] .- truth[1:end-1]
    roc_pred = pred[2:end] .- pred[1:end-1]
    mae(roc_truth, roc_pred)
end

function perc_mae(truth::T, pred::T) where {T <: Matrix}
    weight = 1 / size(truth, 1)
    abs.(pred .- truth) |>
        x -> sum(eachrow(x)) .* weight |>
        x -> mean(x)
end

function perc_mae(truth::T, pred::T) where {T <: Vector}
    abs.(pred .- truth) ./ truth |>
        x -> mean(x)
end

function perc_roc(truth::T, pred::T) where {T <: Matrix}
    roc_truth = truth[2:end, :] .- truth[1:end-1, :]
    roc_pred = pred[2:end, :] .- pred[1:end-1, :]
    perc_mae(roc_truth, roc_pred)
end

function perc_roc(truth::T, pred::T) where {T <: Vector}
    roc_truth = truth[2:end] .- truth[1:end-1]
    roc_pred = pred[2:end] .- pred[1:end-1]
    perc_mae(roc_truth, roc_pred)
end

import Flux
findnearest(A::Vector,t) = findmin(abs.(A .- t))[2]


function retroactive_loss(prediction::Matrix, prob, solver; loss=Flux.Losses.mae)
    """
    Prediction for time is in the top row of the 
    prediction matrix.
    """

    saveat = 10 .^ prediction[begin, :]
    sol = solve(prob, saveat, solver=solver)
    inds = saveat .|> x->findnearest(sol.t, x)
    true_solution = hcat(sol.u[inds]...) |> x -> (x .- minimum(x)) ./ (maximum(x) - minimum(x))
    true_solution = eachcol(true_solution) .|> x->x ./ sum(abs.(x))
    
    loss(hcat(true_solution...), prediction[2:end, :]) / size(true_solution, 2)
end

function physical_error(truth::T, pred::T) where {T <: Matrix}
    truth_sums = 2 .^ truth |> eachcol .|> sum
    pred_sums = 2 .^ pred |> eachcol .|> sum
    mean(pred_sums .- truth_sums[begin])
end


function normalize_each_column(X::Matrix)
    # here I add a small buffer so I can take the log later on
    X |> x -> (eachcol(x .+ abs(minimum(x))*1.01) .|> x->(x./sum(x))) |> x->hcat(x...)
end

function filter_to_significant_concentration(X::Matrix, transform=:log10; indices_only=false)
    if transform == :log10
        threshold = log10(1e-20)
    elseif transform == :log2
        threshold = log2(1e-20)
    end

    if indices_only
        any.(x->x>=threshold,eachrow(X))
    else
        X[any.(x->x>=threshold,eachrow(X)),:]
    end
end

function sum_columns(X::Matrix, transform)
    if transform == :log10
        10 .^ X |> eachcol .|> sum
    elseif transform == :log2
        2 .^ X |> eachcol .|> sum
    else
        X |> eachcol .|> sum
    end
end