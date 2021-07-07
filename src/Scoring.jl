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
    abs.(pred .- truth) ./ truth |>
        x -> sum(eachrow(x)) |>
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
