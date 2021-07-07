using ReservoirComputing, Surrogates


rates_set_lower_bound = [1e-17, 0.5, 10, 0.5, 2.,  1e2]
rates_set_upper_bound = [1., 0.5, 100, 1.5, 10.,  1e4]

parameter_samples = sample(30, rates_set_lower_bound, rates_set_upper_bound, SobolSample())


res_size = 1500
radius = .8
degree = 1200
activation = tanh
alpha = .9
sigma = .1
nla_type = NLADefault()
extended_states = false
beta = 0.000001

resulting_weights = []

