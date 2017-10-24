
import ../random/random
import ../tensor
import layer

type
   GruLayer* = ref object of Layer
      hidden*, reset_weights, reset_hidden_weights,
         update_weights, update_hidden_weights, hhat_weights: Tensor

method open*(self: GruLayer) =
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

method close*(self: GruLayer) =
   self.values                = nil
   self.weights               = nil
   self.hidden                = nil
   self.reset_weights         = nil
   self.reset_hidden_weights  = nil
   self.update_weights        = nil
   self.update_hidden_weights = nil
   self.hhat_weights          = nil

method forward*(self: GruLayer, inputs: Tensor) =
   # gather data from previous layer
   self.values.set(0)
   inputs.spread(self.weights, self.values)

   # move values to previous timestep
   # NB this feels wrong; seems like a gate should control this
   self.hidden.set(self.values)

   # calculate reset gates
   self.scratch[0].set_mul(self.reset_weights, self.values)
   self.scratch[1].set_mul(self.reset_hidden_weights, self.hidden)
   self.scratch[0].add(self.scratch[1])
   self.scratch[0].sigmoid

   # calculate update gates
   self.scratch[1].set_mul(self.update_weights, self.values)
   self.scratch[2].set_mul(self.update_hidden_weights, self.values)
   self.scratch[1].add(self.scratch[2])
   self.scratch[1].sigmoid

   # calculate hhat
   self.scratch[2].set_mul(self.hidden, self.scratch[0])
   self.scratch[2].mul(self.hhat_weights)
   self.scratch[2].add(self.values)
   self.scratch[2].tanh

   # calculate final output values
   self.values.set(1)
   self.values.sub(self.scratch[1])
   self.values.mul(self.scratch[2])
   self.scratch[0].set_mul(self.scratch[1], self.hidden)
   self.values.add(self.scratch[0])

method gradient*(self: GruLayer, inputs, deltas, total: Tensor) =
   # run inputs * weights to find the amount of error 
   self.scratch[0].set(0)
   deltas.spread(self.weights, self.scratch[0], self.values.len-1)
   total.add(self.scratch[0])

method private_gradient*(self: GruLayer, inputs, deltas, total: Tensor) =
   self.scratch[0].set(self.values)
   self.scratch[0].tanh_deriv
   self.scratch[0].mul(deltas)

   # reset and update gates have identical derivatives
   self.scratch[1].set_mul(self.values, deltas)
   self.scratch[2].set_mul(self.values, self.hidden)

   # pack gate derivatives in output
   total.add(self.scratch[1], 0, 0, self.value_count)
   total.add(self.scratch[2], 0, self.value_count, self.value_count)
   total.add(self.scratch[1], 0, self.value_count*2, self.value_count)
   total.add(self.scratch[2], 0, self.value_count*3, self.value_count)

   # calculating the new reset gate values
   self.scratch[0].set_mul(self.scratch[2], self.hidden)
   self.scratch[0].add(self.scratch[1])
   self.scratch[0].tanh

   # then combine to create derivatives for last state's contribution
   self.scratch[0].mul(self.hidden)
   self.scratch[0].mul(deltas)

   # pack hhat derivatives in output
   total.add(self.scratch[0], 0, self.value_count*4, self.value_count)

method propagate*(self: GruLayer, updates: Tensor) =
   self.weights.sub(updates)

method private_propagate*(self: GruLayer, updates: Tensor) =
   self.reset_weights.sub(updates, 0, 0, self.value_count)
   self.reset_hidden_weights.sub(updates, 0, self.value_count, self.value_count)
   self.update_weights.sub(updates, 0, self.value_count*2, self.value_count)
   self.update_hidden_weights.sub(updates, 0, self.value_count*3, self.value_count)
   self.hhat_weights.sub(updates, 0, self.value_count*4, self.value_count)

method randomize_weights*(self: GruLayer, rng: Random) =
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

