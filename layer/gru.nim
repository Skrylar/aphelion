
import ../random/random
import ../tensor

type
   GruLayer* = ref object
      values, weights, hidden, reset_weights, reset_hidden_weights,
         update_weights, update_hidden_weights, hhat_weights: Tensor

      scratch: array[0..2, Tensor]
      input_count, value_count, private_weight_count: int

method open*(self: GruLayer) {.base.} =
   self.values         =  make_tensor(self.value_count)
   assert(self.values  != nil)
   self.weights        =  make_tensor(self.value_count * self.input_count)
   assert(self.weights != nil)
   self.hidden         =  make_tensor(self.value_count)
   assert(self.hidden  != nil)
   self.reset_weights  =  make_tensor(self.value_count)
   assert(self.reset_weights         != nil)
   self.reset_hidden_weights         =  make_tensor(self.value_count)
   assert(self.reset_hidden_weights  != nil)
   self.update_weights               =  make_tensor(self.value_count)
   assert(self.update_weights        != nil)
   self.update_hidden_weights        =  make_tensor(self.value_count)
   assert(self.update_hidden_weights != nil)
   self.hhat_weights                 =  make_tensor(self.value_count)
   assert(self.hhat_weights          != nil)

method close*(self: GruLayer) {.base.} =
   self.values                = nil
   self.weights               = nil
   self.hidden                = nil
   self.reset_weights         = nil
   self.reset_hidden_weights  = nil
   self.update_weights        = nil
   self.update_hidden_weights = nil
   self.hhat_weights          = nil

method forward*(self: GruLayer, inputs: Tensor) {.base.} =
   self.hidden.set(self.values)

   inputs.set_spread(self.weights, self.values)

   for i in 0..2:
      assert(self.scratch[i] != nil)

   # calculate value of the reset neuron input
   self.scratch[0].set_mul(self.reset_weights, self.values)
   self.scratch[1].set_mul(self.reset_hidden_weights, self.hidden)

   self.scratch[0].add(self.scratch[1], self.hidden.len);
   self.scratch[0].tanh(self.hidden.len);

   # calculate value of the update neuron input
   self.scratch[1].set_mul(self.update_weights, self.values)
   self.scratch[2].set_mul(self.update_hidden_weights, self.hidden)

   self.scratch[1].add(self.scratch[2], self.hidden.len)
   self.scratch[1].tanh(self.hidden.len)

   # var hhat = Curves.Tanh(Values[i] + (Weights[x++] * (reset * Hidden[i])));
   self.scratch[2].set_mul(self.scratch[0], self.hidden)
   self.scratch[2].mul(self.hhat_weights)
   self.scratch[2].add(self.values)

   self.scratch[0].set(1.0)
   self.scratch[0].sub(self.scratch[1])
   self.scratch[2].mul(self.scratch[0])

   self.values.set_mul(self.scratch[1], self.hidden)
   self.values.add(self.scratch[2])

method gradient*(self: GruLayer, inputs, deltas, total: Tensor) {.base.} =
   # get the big stuff
   self.scratch[0].mul_set(deltas, inputs)
   total.add(self.scratch[0])

method private_gradient*(self: GruLayer, inputs, deltas, total: Tensor) {.base.} =
   # calculate amount of change necessary
   self.scratch[0].set self.values
   self.scratch[0].tanh_deriv
   self.scratch[0].mul deltas

   # delta * values
   self.scratch[1].set_add self.scratch[0], self.values
   # delta * hidden
   self.scratch[2].set_add self.scratch[0], self.hidden

   # pack reset gates
   total.add(self.scratch[1], 0, 0, self.value_count)
   total.add(self.scratch[2], self.value_count, 0, self.value_count)

   # pack update gates
   total.add(self.scratch[1], self.value_count*2, 0, self.value_count)
   total.add(self.scratch[2], self.value_count*3, 0, self.value_count)

   # calculate weight delta to our hidden state
   self.scratch[2].set_mul self.hidden, self.scratch[2]
   self.scratch[1].set_mul self.values, self.scratch[1]
   self.scratch[1].add self.scratch[2], self.value_count
   self.scratch[1].tanh self.value_count
   self.scratch[1].mul self.scratch[0], self.value_count
   self.scratch[1].mul self.hidden

   # pack hidden reset weight
   total.add(self.scratch[1], self.value_count*4, 0, self.value_count)

method propagate*(self: GruLayer, updates: Tensor) {.base.} =
   self.weights.sub(updates)

method private_propagate*(self: GruLayer, updates: Tensor) {.base.} =
   self.reset_weights.sub(updates, 0, 0, self.value_count)
   self.reset_hidden_weights.sub(updates, 0, self.value_count, self.value_count)
   self.update_weights.sub(updates, 0, self.value_count*2, self.value_count)
   self.update_hidden_weights.sub(updates, 0, self.value_count*3, self.value_count)
   self.hhat_weights.sub(updates, 0, self.value_count*4, self.value_count)

method randomize_weights*(self: GruLayer, rng: Random) {.base.} =
   for i in 0..(self.value_count - 1):
      self.weights.set_at(i, rng.next_float)
      self.reset_weights.set_at(i, rng.next_float)
      self.reset_hidden_weights.set_at(i, rng.next_float)
      self.update_weights.set_at(i, rng.next_float)
      self.update_hidden_weights.set_at(i, rng.next_float)
      self.hhat_weights.set_at(i, rng.next_float)

proc make_gru_layer*(inputs, outputs: int): GruLayer =
   result = GruLayer()
   result.input_count = inputs
   result.value_count = outputs
   result.private_weight_count = outputs * 5
   result.open

