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

function retroactive_loss(prediction::Matrix, prob, solver; loss=Flux.Losses.mae)
    """
    Prediction for time is in the top row of the 
    prediction matrix.
    """
    saveat = prediction[begin, :]
    sol = solve(prob, saveat, solver=solver())
    true_solution = hcat(sol.u...)
    loss(true_solution, prediction[2:end, :])
end
