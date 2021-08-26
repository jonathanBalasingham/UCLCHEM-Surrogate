using DrWatson
@quickactivate "UCLCHEM Surrogate"
using Flux
include(srcdir("EchoStateNetwork.jl"))

# Dummy data to make ESN fit a straight line
X = ones(3, 100)
y = ones(3, 100)

input_dimension = size(X, 1)
output_dimension = size(y, 1)

reservoir_size = 50

# Create the network and train on  X, y
# train! expects an array of data so X and
# y must be boxed.
esn = ESN.EchoStateNetwork{Float64, ESN.EchoStateReservoir{Float64}}(input_dimension, reservoir_size, output_dimension);

regularization = 1e-9
ESN.train!(esn, [X], [y], regularization)

# Give the ESN the first sample and 
# let it predict the rest.
prediction = ESN.predict!(esn, X[:, begin:begin], 99)

# Measure the error
error = Flux.Losses.mae(prediction, y);
@info "MAE of: $error"