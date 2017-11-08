
import ../random/random
import ../tensor
import layer

type
   TanhLayer* = ref object of Layer

method forward*(self: TanhLayer; inputs: Tensor; scratch: ScratchSet) =
  let n = self.values.len
  let a = n*self.input_count

  # clear all values
  self.values.set(0)
  # run weight*value for each neuron
  #echo inputs.len, "*", self.weights.len, "=", self.values.len
  inputs.spread(self.weights, self.values, self.input_count-1, n-1)
  # add linear biases
  self.values.add(self.weights, 0, a, n)
  # apply curve
  self.values.tanh

method gradient*(self: TanhLayer; inputs, deltas, total: Tensor; scratch: ScratchSet) =
  let n = self.values.len
  let a = n*self.input_count

  # clear the tensor
  scratch[0].set(0)
  scratch[1].set(self.values)
  scratch[1].tanh_deriv
  # weights * deltas to determine error contribution
  deltas.spread(self.weights, scratch[0], self.input_count-1, n-1)
  # push deltas to our bias nodes
  total.add(scratch[0], a, 0, n)
  # multiply by derivative of curves, apply those to our input weights
  scratch[0].mul scratch[1]
  total.add(scratch[0], 0, 0, a)

proc make_tanh_layer*(inputs, outputs: int): TanhLayer =
   result = TanhLayer()
   assert result != nil
   result.internal_weights = 1
   result.input_count = inputs
   result.values = make_tensor(outputs)
   assert result.values != nil
   result.weights = make_tensor((outputs * result.internal_weights) + (outputs * result.input_count))
   assert result.weights != nil