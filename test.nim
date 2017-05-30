
import tensor
import criterion/meansquarederrors
import random/cmwc
import layer/linear
import layer/tanh

var randomizer = Cmwc()
randomizer.seed(1337)
assert randomizer != nil

var this_error: array[0..1, Tensor]
var total_error: array[0..1, Tensor]
for i in 0..1:
   this_error[i] = make_tensor(3*3)
   this_error[i].set(0.0)

   total_error[i] = make_tensor(3*3)
   total_error[i].set(0.0)

var scratch: array[0..2, Tensor]
for i in 0..high(scratch):
   scratch[i] = make_tensor(3*3)

# create the network
var input = make_tensor(3)
assert input != nil

var goals = make_tensor(3)
assert goals != nil

goals.set_at(0, 0.0)
goals.set_at(1, 1.0)
goals.set_at(2, 0.0)

var layer = make_linear_layer(3, 3);
assert layer != nil

var layer2 = make_tanh_layer(3, 3);
assert layer2 != nil

# initialize
layer.randomize_weights randomizer
layer2.randomize_weights randomizer

for i in 0..4:
   # forward pass
   layer.forward input
   layer2.forward layer.values

   # calculate errors
   var loss = mse(scratch[0], layer2.values, goals)
   echo("Loss ", loss)

   # work backwards
   
   # start from derivatives of the goal function
   this_error[0].mse_derivs layer2.values, goals

   # apply derivatives
   layer2.gradient layer.values, this_error[0], total_error[0]

   # backpropagate
   layer2.propagate total_error[0]

   # apply derivatives to private weights
   layer2.private_gradient layer.values, this_error[1], total_error[0]

   # backpropagate private weights
   layer2.private_propagate this_error[1]

   set(total_error[0], 0.0)
   set(total_error[1], 0.0)

