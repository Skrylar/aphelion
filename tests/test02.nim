
import math
import tensor
import criterion/meansquarederrors
import random/cmwc
import layer/linear
import simplenetwork
import backpropagator
import backpropagation/adam

var randomizer = Cmwc()
randomizer.seed(1337)
assert randomizer != nil

# create the network
var input = make_tensor(3)
assert input != nil

var goals = make_tensor(3)
assert goals != nil

var net = make_simple_network(input)
#var bp = GruHypertrainer()
var bp = AdamBackpropagator()
var mse = MseCriterion()

net.add_linear_layer(3)
net.add_linear_layer(3)
net.add_linear_layer(3)
net.add_linear_layer(3)

net.auto_scratch_tensors
net.randomize_weights(randomizer)

bp.init(net)

input[0] = 0.0
input[1] = 0.0
input[2] = 1.0

goals[0] = 1.0
goals[1] = 1.0
goals[2] = 0.0

var best = 999999999.0

for i in 0..10000:
  net.forward
  let loss = mse.loss(net.scratch, net.layers[net.layers.high], goals)
  if loss.classify == fcNaN:
    break
  echo "Loss: ", loss
  if loss < best: best = loss
  bp.clear_gradients
  bp.backward goals, mse, net, net.scratch
  bp.propagate net

  #bp.sgd(0.00005, net, net.scratch)

echo "Loss: ", mse.loss(net.scratch, net.layers[net.layers.high], goals)
echo "Best: ", best
net.forward
echo repr net.layers[net.layers.high]
