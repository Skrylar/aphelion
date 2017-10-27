
import criterion/criterion
import simplenetwork
import gradientmap
import tensor
import layer/layer

type
  Backpropagator* = ref object {.inheritable.}
    errors*, total_errors*: GradientMap

method create_gradient_map*(self: Backpropagator, net: SimpleNetwork) {.base.} =
  ## Prepares the backpropagator to receive per-layer error values from
  ## a given network.
  setLen(self.errors       , net.layers.len)
  setLen(self.total_errors , net.layers.len)

  template hackysack(x, y, z: untyped) =
    if (x[y] == nil) or (x[y].len != z):
      x[y] = make_tensor(z)

  for i in 0..net.layers.high:
    hackysack(self.errors        , i , net.layers[i].values.len)
    hackysack(self.total_errors  , i , net.layers[i].weights.len)

proc init_backpropagator*(self: Backpropagator; net: SimpleNetwork) =
  newseq(self.errors, 0)
  newseq(self.total_errors, 0)
  self.create_gradient_map(net)

method init*(self: Backpropagator; net: SimpleNetwork) {.base.} =
  init_backpropagator self, net

proc clear_gradients*(self: Backpropagator) =
  ## Clears any gradients that have accumulated between calls to
  ## backwards.
  for layer in self.total_errors:
    layer.set(0)

proc backward*(self: Backpropagator, goals: Tensor,
  criterion: Criterion, net: SimpleNetwork, scratch: ScratchSet) =
    let last_layer_index = net.layers.high

    # derive initial output score
    criterion.derive(net.layers[last_layer_index],
      goals, self.errors[last_layer_index])

    # TODO separate output layer to hold these derivatives, since they
    # are derived from the final layer's outputs; we currently just
    # cobble over the last layer to store outputs

    # perform gradient calculation for front layer
    net.layers[last_layer_index].gradient(
      net.layers[last_layer_index - 1].values,
      self.errors[last_layer_index],
      self.total_errors[last_layer_index],
      scratch)

    # calculate gradients of middle layers
    for x in countdown(last_layer_index - 1, 1):
      # now moderate gradients by next layer's weight values
      self.errors[x].set(0)
      despread(self.errors[x], net.layers[x+1].weights, net.layers[x+1].values, net.layers[x + 1].weights.len, net.layers[x+1].values.len)

      # perform gradient calculation
      net.layers[x].gradient(
        net.layers[x - 1].values,
        self.errors[x],
        self.total_errors[x],
        scratch)

    self.errors[0].set(0)
    despread(self.errors[0], net.layers[1].weights, net.layers[1].values, net.layers[1].weights.len, net.layers[1].values.len)
    
    # calculate gradients on the final layer
    net.layers[0].gradient(net.inputs,
       self.errors[0],
       self.total_errors[0],
       scratch)

proc sgd*(self: Backpropagator, rate: float, net: SimpleNetwork; scratch: ScratchSet) =
   # TODO don't multiply scratch space that isn't used by weights
   for i in 0..net.layers.high:
      scratch[0].set(self.total_errors[i])
      scratch[0].mul(rate)
      net.layers[i].propagate(scratch[0])

