
import layer
import ../random/random
import ../tensor

type
   LinearLayer* = ref object of Layer
      private_weights: Tensor

method open*(self: LinearLayer) =
   self.values = make_tensor(self.value_count)
   assert(self.values != nil)
   self.weights = make_tensor(self.value_count * self.input_count)
   assert(self.weights != nil)
   self.private_weights = make_tensor(self.value_count);
   assert(self.private_weights != nil)

method close*(self: LinearLayer) =
   self.values          = nil
   self.weights         = nil
   self.private_weights = nil

method forward*(self: LinearLayer, inputs: Tensor) =
   inputs.spread(self.weights, self.values)
   self.values.divide(float32(inputs.len))
   self.values.add(self.private_weights)

method gradient*(self: LinearLayer, inputs, deltas, total: Tensor) =
   # get the big stuff
   self.scratch[0].set_mul(deltas, inputs)
   total.add(deltas)

method private_gradient*(self: LinearLayer, inputs, deltas, total: Tensor) =
   # fix up the private deltas
   total.add(self.private_weights)

method propagate*(self: LinearLayer, updates: Tensor) =
   self.weights.sub(updates)

method private_propagate*(self: LinearLayer, updates: Tensor) =
   self.private_weights.sub(updates)

method randomize_weights*(self: LinearLayer, rng: Random) =
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

