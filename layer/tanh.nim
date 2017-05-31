
import ../random/random
import ../tensor
import layer

type
   TanhLayer* = ref object of Layer

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
   self.scratch[0].set_mul(deltas, inputs)
   self.scratch[0].tanh_deriv
   total.add(self.scratch[0])

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

