
import tensor
import criterion/meansquarederrors
import random/cmwc
import layer/linear
import layer/tanh
import simplenetwork

var randomizer = Cmwc()
randomizer.seed(1337)
assert randomizer != nil

var net = make_simple_network(3)

net.add_linear_layer(3)
net.add_tanh_layer(3)

net.auto_scratch_tensors
net.create_gradient_map
net.randomize_weights(randomizer)

# create the network
var input = make_tensor(3)
assert input != nil

var goals = make_tensor(3)
assert goals != nil

input[0] = 1.0
input[1] = 1.0
input[2] = 1.0

goals[0] = 0.0
goals[1] = 1.0
goals[2] = 0.0

net.forward input

