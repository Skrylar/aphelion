
import ../random/random
import ../tensor

type
   TanhLayer* = ref object
      values, weights: Tensor
      scratch: array[0..2, Tensor]
      input_count, value_count, private_weight_count: int

method open*(self: TanhLayer) {.base.} =
   self.values = make_tensor(self.value_count)
   assert(self.values != nil)
   self.weights = make_tensor(self.value_count * self.input_count)
   assert(self.weights != nil)

method close*(self: TanhLayer) {.base.} =
   self.values  = nil
   self.weights = nil

method forward*(self: TanhLayer, inputs: Tensor) {.base.} =
   inputs.set_spread(self.weights, self.values)
   self.values.divide(float32(inputs.len))
   self.values.tanh

method gradient*(self: TanhLayer, inputs, deltas, total: Tensor) {.base.} =
   # get the big stuff
   deltas.mul(inputs)
   deltas.tanh_deriv
   total.add(deltas)

method private_gradient*(self: TanhLayer, inputs, deltas, total: Tensor) {.base.} =
  discard

method propagate*(self: TanhLayer, updates: Tensor) {.base.} =
   self.weights.sub(updates)

method private_propagate*(self: TanhLayer, updates: Tensor) {.base.} =
  discard

method randomize_weights*(self: TanhLayer, rng: Random) {.base.} =
   for i in 0..(self.weights.len - 1):
      self.weights.set_at(i, rng.next_float);

proc make_tanh_layer*(inputs, outputs: int): TanhLayer =
   result = TanhLayer()
   result.input_count = inputs
   result.value_count = outputs
   result.private_weight_count = 0
   result.open

