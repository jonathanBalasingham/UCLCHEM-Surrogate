include("Constants.jl")

get_u0(x) = x[:, begin]
transform(x; u0=get_u0(x)) = hcat((eachrow(x) ./ u0)...)' |> Matrix
transform_back(x; u0) = hcat((eachrow(x) .* u0)...)' |> Matrix

bottom = filter(x->x>0,true_dark_cloud_lower) |> minimum |> log10 |> abs |> x->round(x)+1 
r(x) = replace(log10.(x) .+ bottom, -Inf=>0.0)
inverse_r(x) = 10 .^ (x .- bottom)
