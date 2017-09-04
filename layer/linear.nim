
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
   self.values.set(0)
   inputs.spread(self.weights, self.values)
   # NB: Not sure we actually need to do this
   #self.values.divide(float32(inputs.len))
   self.values.add(self.private_weights)

method gradient*(self: LinearLayer, inputs, deltas, total: Tensor) =
   # run inputs * weights to find the amount of error 
   self.scratch[0].set(0)
   deltas.spread(self.weights, self.scratch[0], self.values.len-1)
   total.add(self.scratch[0])

method private_gradient*(self: LinearLayer, inputs, deltas, total: Tensor) =
   # update linear bias weight by the neuron's delta directly
   total.add(deltas)

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

