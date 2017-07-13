
import tensor
import criterion/meansquarederrors
import random/cmwc
import layer/linear
import layer/tanh
import simplenetwork
import backpropagator

var randomizer = Cmwc()
randomizer.seed(1337)
assert randomizer != nil

# create the network
var input = make_tensor(3)
assert input != nil

var goals = make_tensor(3)
assert goals != nil

var net = make_simple_network(input)
var bp = make_backpropagator()
var mse = MseCriterion()

net.add_linear_layer(10)
net.add_tanh_layer(3)
net.add_linear_layer(3)

net.auto_scratch_tensors
net.randomize_weights(randomizer)

bp.init(net)

input[0] = 0.0
input[1] = 1.0
input[2] = 0.0

goals[0] = 0.0
goals[1] = 1.0
goals[2] = 0.0

for i in 0..5:
  net.forward
  echo "Loss: ", mse.loss(net.layers[net.layers.high], goals)
  bp.clear_gradients
  bp.backward(goals, mse, net)
  bp.sgd(0.00005, net)

echo "Loss ", mse.loss(net.layers[net.layers.high], goals)
net.forward
echo repr net.layers[net.layers.high]

