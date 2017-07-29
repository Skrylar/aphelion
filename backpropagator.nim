
import criterion/criterion
import simplenetwork
import gradientmap
import tensor
import layer/layer

type
   Backpropagator* = ref object {.inheritable.}
      public_errors*, private_errors*: GradientMap
      total_public_errors*, total_private_errors*: GradientMap

method create_gradient_map*(self: Backpropagator, net: SimpleNetwork) {.base.} =
   ## Prepares the backpropagator to receive per-layer error values from
   ## a given network.
   setLen(self.public_errors        , net.layers.len)
   setLen(self.private_errors       , net.layers.len)
   setLen(self.total_public_errors  , net.layers.len)
   setLen(self.total_private_errors , net.layers.len)

   template hackysack(x, y, z: untyped) =
      if (x[y] == nil) or (x[y].len != z):
         x[y] = make_tensor(z)

   for i in 0..net.layers.high:
      hackysack(self.public_errors        , i , net.layers[i].value_count)
      hackysack(self.private_errors       , i , net.layers[i].value_count)
      hackysack(self.total_public_errors  , i , net.layers[i].value_count)
      hackysack(self.total_private_errors , i , net.layers[i].value_count)

method init*(self: Backpropagator; net: SimpleNetwork) {.base.} =
   newseq(self.public_errors, 0)
   newseq(self.private_errors, 0)
   newseq(self.total_public_errors, 0)
   newseq(self.total_private_errors, 0)
   self.create_gradient_map(net)

proc clear_gradients*(self: Backpropagator) =
   ## Clears any gradients that have accumulated between calls to
   ## backwards.
   for layer in self.total_public_errors:
      layer.set(0)
   for layer in self.total_private_errors:
      layer.set(0)

proc backward*(self: Backpropagator, goals: Tensor,
   criterion: Criterion, net: SimpleNetwork) =
      let last_layer_index = net.layers.high

      # derive initial output score
      criterion.derive(net.layers[last_layer_index],
         goals, self.public_errors[last_layer_index])

      # perform gradient calculation for front layer
      #net.layers[last_layer_index].gradient(
         #net.layers[last_layer_index - 1].values,
         #self.public_errors[last_layer_index],
         #self.total_public_errors[last_layer_index])

      #net.layers[last_layer_index].private_gradient(
         #net.layers[last_layer_index - 1].values,
         #self.public_errors[last_layer_index],
         #self.total_private_errors[last_layer_index])

      # calculate gradients of middle layers
      for x in countdown(last_layer_index - 1, 1):
         # perform gradient calculation
         net.layers[x].gradient(
            net.layers[x - 1].values,
            self.public_errors[x],
            self.total_public_errors[x])

         net.layers[x].private_gradient(
            net.layers[x - 1].values,
            self.public_errors[x],
            self.total_private_errors[x])

         # now moderate gradients by next layer's weight values
         weight_sum(net.scratch[0], net.layers[x + 1].weights, net.layers[x + 1].weights.len, net.layers[x].values.len)
         self.public_errors[x].mul(net.scratch[0], net.layers[x].values.len)
      
      # calculate gradients on the final layer
      net.layers[0].gradient(net.inputs,
         self.public_errors[0],
         self.total_public_errors[0])

      net.layers[0].private_gradient(net.inputs,
         self.private_errors[0],
         self.total_public_errors[0])

      if net.layers.high > 1:
         weight_sum(net.scratch[0], net.layers[1].weights, net.layers[1].weights.len, net.layers[0].values.len)
         self.public_errors[0].mul(net.scratch[0], net.layers[0].values.len)

proc sgd*(self: Backpropagator, rate: float, net: SimpleNetwork) =
   # TODO don't multiply scratch space that isn't used by weights
   for i in 0..net.layers.high:
      net.scratch[0].set(self.total_public_errors[i])
      net.scratch[0].mul(rate)
      net.layers[i].propagate(net.scratch[0])

      net.scratch[0].set(self.total_private_errors[i])
      net.scratch[0].mul(rate)
      net.layers[i].private_propagate(net.scratch[0])

