
import ../random/random
import ../tensor

type
   LinearLayer* = ref object
      values, weights, private_weights: Tensor
      input_count, value_count, private_weight_count: int

method open*(self: LinearLayer) {.base.} =
   self.values = make_tensor(self.value_count)
   assert(self.values != nil)
   self.weights = make_tensor(self.value_count * self.input_count)
   assert(self.weights != nil)
   self.private_weights = make_tensor(self.value_count);
   assert(self.private_weights != nil)

method close*(self: LinearLayer) {.base.} =
   self.values          = nil
   self.weights         = nil
   self.private_weights = nil

method forward*(self: LinearLayer, inputs: Tensor) {.base.} =
   inputs.set_spread(self.weights, self.values)
   self.values.divide(float32(inputs.len))
   self.values.add(self.private_weights)

method gradient*(self: LinearLayer, inputs, deltas: Tensor) {.base.} =
   # get the big stuff
   deltas.mul(inputs)

method private_gradient*(self: LinearLayer, inputs, deltas: Tensor) {.base.} =
   # fix up the private deltas
   deltas.add(self.private_weights)

method propagate*(self: LinearLayer, updates: Tensor) {.base.} =
   self.weights.sub(updates)

method private_propagate*(self: LinearLayer, updates: Tensor) {.base.} =
   self.private_weights.sub(updates)

method randomize_weights*(self: LinearLayer, rng: Random) {.base.} =
   for i in 0..(self.weights.len - 1):
      self.weights.set_at(i, rng.next_float);

   for i in 0..(self.private_weights.len - 1):
      self.private_weights.set_at(i, rng.next_float)

proc make_linear_layer*(inputs, outputs: int): LinearLayer =
   result = LinearLayer()
   result.input_count = inputs
   result.value_count = outputs
   result.private_weight_count = outputs
   result.open

