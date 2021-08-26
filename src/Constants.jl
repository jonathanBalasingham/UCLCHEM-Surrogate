include("GasPhaseNetwork.jl")


const tspan = (0., 10^7 * 365. * 24. * 3600.)
#  zeta, omega, T, F_UV, A_v, E, density
const rates_set_lower_bound = [1e-17, 0.5, 10, 1., 10., 1e2]
const rates_set_upper_bound = [1e-14, 0.5, 300, 1., 10., 1e6]

const dark_cloud_upper = get_rates(rfp, Parameters(rates_set_upper_bound...))
const dark_cloud_lower = get_rates(rfp, Parameters(rates_set_lower_bound...))
const true_dark_cloud_lower = [min(a,b) for (a,b) in zip(dark_cloud_lower, dark_cloud_upper)]
const true_dark_cloud_upper = [max(a,b) for (a,b) in zip(dark_cloud_lower, dark_cloud_upper)]
