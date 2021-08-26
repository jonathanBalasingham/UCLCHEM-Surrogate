using Plots
include(srcdir("Scoring.jl"))

function visualize(sol::ChemicalNetworkSolution; species=nothing, size=(1300,800), take_every_nth=1)
    if isnothing(species)
        plot(sol.t[1:take_every_nth:end] ./ (3600 * 24 * 365), 
            log10.(transpose(reduce(hcat, sol.u[1:take_every_nth:end]))), 
            size=size, 
            labels=permutedims(sol.species),
            xticks=([10^0, 10,10^2,10^3,10^4,10^5,10^6],["10⁰", "10","10²","10³","10⁴","10⁵","10⁶"]),
            yticks=([10^-15, 10^-13, 10^-11, 10^-9, 10^-7, 10^-5, 10^-3, 10^-1, 10^1],["10⁻¹⁵", "10⁻¹³", "10⁻¹¹", "10⁻⁹", "10⁻⁷", "10⁻⁵", "10⁻³", "10⁻¹", "10¹"]),
            xaxis=:log10,
            legend=:outertopright)
    else
        inds =  indexin(species, sol.species)
        data = transpose(reduce(hcat, sol.u[1:take_every_nth:end]))[:,inds]
        data[data .<= 0.0] .= 1e-60
        
        plot(sol.t[1:take_every_nth:end] ./ (3600 * 24 * 365), 
             log10.(data),
             #size=size, 
             labels=permutedims(sol.species[inds]),
             xticks=([10^0, 10,10^2,10^3,10^4,10^5,10^6],["10⁰", "10","10²","10³","10⁴","10⁵","10⁶"]),
             yticks=(log10.([10^-15, 10^-13, 10^-11, 10^-9, 10^-7, 10^-5, 10^-3, 10^-1, 10^1]),["10⁻¹⁵", "10⁻¹³", "10⁻¹¹", "10⁻⁹", "10⁻⁷", "10⁻⁵", "10⁻³", "10⁻¹", "10¹"]),
             xaxis=:log10,
             legend=:outertopright)
    end
    xlims!(1, 10^7)
    ylims!(log10.((10^-15, 10)))
    xlabel!("Time / Years")
    ylabel!("Xₛₚₑ")
end

function visualize(interpolation::Matrix, 
                  prediction::Matrix, 
                  solution::Matrix, 
                  timepoints::Vector; full_species, plotted_species, size=(1300,800))
    indx =  indexin(plotted_species, full_species)
    log2_to_log10(x) = log10.(2 .^ x)
    s = reshape(full_species[indx], 1, :)
    p1 = plot(timepoints ./ (3600 * 24 * 365), 
              log2_to_log10(solution[indx, :])',
              labels=s,
              xticks=([10^0, 10,10^2,10^3,10^4,10^5,10^6],["10⁰", "10","10²","10³","10⁴","10⁵","10⁶"]),
              ylabel="Xₛₚₑ",
              xlabel="Time / Years",
              xaxis=:log10,
              title="Ground Truth",
              legend=:outertopright)
              xlims!(1, 10^7)
              yticks!([-15, -13, -11, -9, -7, -5, -3, -1, 1],["10⁻¹⁵", "10⁻¹³", "10⁻¹¹", "10⁻⁹", "10⁻⁷", "10⁻⁵", "10⁻³", "10⁻¹", "10¹"])
              ylims!(-15, 1)
    p2 = plot(timepoints ./ (3600 * 24 * 365), 
              log2_to_log10(prediction[indx, :])',
              labels=s,
              xticks=([10^0, 10,10^2,10^3,10^4,10^5,10^6],["10⁰", "10","10²","10³","10⁴","10⁵","10⁶"]),
              ylabel="Xₛₚₑ",
              xlabel="Time / Years",
              xaxis=:log10,
              title="Trained",
              legend=:outertopright)
              xlims!(1, 10^7)
              yticks!([-15, -13, -11, -9, -7, -5, -3, -1, 1],["10⁻¹⁵", "10⁻¹³", "10⁻¹¹", "10⁻⁹", "10⁻⁷", "10⁻⁵", "10⁻³", "10⁻¹", "10¹"])
              ylims!(-15, 1)

    p3 = plot(timepoints ./ (3600 * 24 * 365), 
              log2_to_log10(interpolation[indx, :])',
              labels=s,
              xticks=([10^0, 10,10^2,10^3,10^4,10^5,10^6],["10⁰", "10","10²","10³","10⁴","10⁵","10⁶"]),
              ylabel="Xₛₚₑ",
              xlabel="Time / Years",
              xaxis=:log10,
              title="Interpolation",
              legend=:outertopright)
              xlims!(1, 10^7)
              yticks!([-15, -13, -11, -9, -7, -5, -3, -1, 1],["10⁻¹⁵", "10⁻¹³", "10⁻¹¹", "10⁻⁹", "10⁻⁷", "10⁻⁵", "10⁻³", "10⁻¹", "10¹"])

              ylims!(-15, 1)


    total_prod = (2 .^ prediction)[indx, :] |> x->sum_columns(x, nothing, false)
    total_interp = (2 .^ interpolation)[indx, :] |> x->sum_columns(x, nothing, false)
    total_gt = (2 .^ solution)[indx, :] |> x->sum_columns(x, nothing, false)

    p4 = plot(timepoints ./ (3600 * 24 * 365), 
              hcat(total_gt, total_prod, total_interp),
              labels=["Ground Truth"  "Prediction"  "Interpolation"],
              ylabel="Total Xₛₚₑ",
              title="Sum of Species Concentration",
              xlabel="Time / Years",
              xticks=([10^0, 10,10^2,10^3,10^4,10^5,10^6],["10⁰", "10","10²","10³","10⁴","10⁵","10⁶"]),
              xaxis=:log10,
              yaxis=:log10,
              legend=:outertopright)
              
    xlims!(1, 10^7)
    plot(p1, p2, p3, p4, layout = (2, 2), size = (1200, 1000))
end