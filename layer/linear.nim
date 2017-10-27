
import layer
import ../random/random
import ../tensor

type
   LinearLayer* = ref object of Layer

method forward*(self: LinearLayer; inputs: Tensor; scratch: ScratchSet) =
  let n = (self.values.len-1)
  let a = n*self.input_count

  # clear all values
  self.values.set(0)
  # run weight*value for each neuron
  inputs.spread(self.weights, self.values, a)
  # add linear biases
  self.values.add(self.weights, 0, a, n)

method gradient*(self: LinearLayer; inputs, deltas, total: Tensor; scratch: ScratchSet) =
  let n = (self.values.len-1)
  let a = n*self.input_count

  # clear the tensor
  scratch[0].set(0)
  # weights * deltas to determine error contribution
  deltas.spread(self.weights, scratch[0], self.values.len-1)
  # add contributions
  total.add(scratch[0], 0, 0, a)
  # also push deltas to update our biases
  total.add(scratch[0], a, 0, n)

proc make_linear_layer*(inputs, outputs: int): LinearLayer =
   result = LinearLayer()
   assert result != nil
   result.internal_weights = 1
   result.input_count = inputs
   result.values = make_tensor(outputs)
   assert result.values != nil
   result.weights = make_tensor((outputs * result.internal_weights) + (outputs * result.input_count))
   assert result.weights != nil

