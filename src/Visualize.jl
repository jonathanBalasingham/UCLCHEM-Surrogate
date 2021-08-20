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

function visualize(interpolation::Matrix, prediction::Matrix, solution::Matrix, timepoints::Vector; species=nothing, size=(1300,800))
    indx = filter_to_significant_concentration(solution, :log2; indices_only=true)
    log2_to_log10(x) = log10(2 .^ x)
    p1 = plot(timepoints ./ (3600 * 24 * 365), 
              log2_to_log10(solution[indx, :])',
              labels=species[indx],
              xticks=([10^0, 10,10^2,10^3,10^4,10^5,10^6],["10⁰", "10","10²","10³","10⁴","10⁵","10⁶"]),
              yticks=([10^-15, 10^-13, 10^-11, 10^-9, 10^-7, 10^-5, 10^-3, 10^-1, 10^1],["10⁻¹⁵", "10⁻¹³", "10⁻¹¹", "10⁻⁹", "10⁻⁷", "10⁻⁵", "10⁻³", "10⁻¹", "10¹"]),
              ylabel="Xₛₚₑ",
              xlabel="Time / Years",
              ylims=(20., 1.),
              xaxis=:log10,
              legend=:outertopright)

    indx = filter_to_significant_concentration(prediction, :log2; indices_only=true)
    p2 = plot(timepoints ./ (3600 * 24 * 365), 
              log2_to_log10(prediction[indx, :])',
              labels=species[indx],
              xticks=([10^0, 10,10^2,10^3,10^4,10^5,10^6],["10⁰", "10","10²","10³","10⁴","10⁵","10⁶"]),
              yticks=([10^-15, 10^-13, 10^-11, 10^-9, 10^-7, 10^-5, 10^-3, 10^-1, 10^1],["10⁻¹⁵", "10⁻¹³", "10⁻¹¹", "10⁻⁹", "10⁻⁷", "10⁻⁵", "10⁻³", "10⁻¹", "10¹"]),
              ylabel="Xₛₚₑ",
              xlabel="Time / Years",
              ylims=(20., 1.),
              xaxis=:log10,
              legend=:outertopright)

    indx = filter_to_significant_concentration(prediction, :log2; indices_only=true)
    p3 = plot(timepoints ./ (3600 * 24 * 365), 
              log2_to_log10(interpolation[indx, :])',
              labels=species[indx],
              xticks=([10^0, 10,10^2,10^3,10^4,10^5,10^6],["10⁰", "10","10²","10³","10⁴","10⁵","10⁶"]),
              yticks=([10^-15, 10^-13, 10^-11, 10^-9, 10^-7, 10^-5, 10^-3, 10^-1, 10^1],["10⁻¹⁵", "10⁻¹³", "10⁻¹¹", "10⁻⁹", "10⁻⁷", "10⁻⁵", "10⁻³", "10⁻¹", "10¹"]),
              ylabel="Xₛₚₑ",
              xlabel="Time / Years",
              ylims=(20., 1.),
              xaxis=:log10,
              legend=:outertopright)

    p4 = plot(timepoints ./ (3600 * 24 * 365), 
              hcat(sum_columns(solution, :log2), sum_columns(prediction, :log2), sum_columns(interpolation, :log2)),
              labels=["Ground Truth", "Prediction", "Interpolation"],
              ylabel="Total Xₛₚₑ",
              xlabel="Time / Years",
              xticks=([10^0, 10,10^2,10^3,10^4,10^5,10^6],["10⁰", "10","10²","10³","10⁴","10⁵","10⁶"]),
              xaxis=:log10,
              legend=:outertopright)

    plot(p1, p2, p3, p4, layout = (2, 2))
end