
import tensor
import random/random
import gradientmap

import layer/layer
import layer/linear
import layer/tanh
import layer/gru

const
   ScratchTensorCount* = 3
   ScratchTensorHigh* = ScratchTensorCount - 1

type
   SimpleNetwork* = ref object
      inputs*: Tensor
      layers*: seq[Layer]

      scratch*: array[0..ScratchTensorHigh, Tensor]

proc make_simple_network*(inputs: Tensor): SimpleNetwork =
   ## Creates and returns a simple network object.
   result = SimpleNetwork()
   newseq(result.layers, 0)
   result.inputs = inputs

proc forward*(self: SimpleNetwork) =
   ## Runs a tensor of values forward through the network.
   if self.layers.len < 1: return
   self.layers[0].forward(self.inputs)
   for i in 1..self.layers.high:
      self.layers[i].forward(self.layers[i - 1].values)

proc randomize_weights*(self: SimpleNetwork, rng: Random) =
   ## Randomizes all weights in the network.
   for layer in self.layers:
      layer.randomize_weights(rng)

proc most_weights*(self: SimpleNetwork): int =
   for layer in self.layers:
      let lx = max(layer.value_count, layer.private_weight_count)
      result = max(result, lx)

proc auto_scratch_tensors*(self: SimpleNetwork) =
   ## Inspects the network, automatically creating scratch memory
   ## tensors and assigning them to layers.
   let m = self.most_weights

   # ensure tensors are in place
   for i in 0..ScratchTensorHigh:
      if (self.scratch[i] == nil) or (self.scratch[i].len < m):
         self.scratch[i] = make_tensor(m)
         assert self.scratch[i] != nil

   # now assign the tensors
   for layer in self.layers:
      for i in 0..ScratchTensorHigh:
         layer.scratch[i] = self.scratch[i]

#_______________________________________________________________________
# Layer adding tools
#_______________________________________________________________________

proc last_output_count(self: SimpleNetwork): int =
   ## Returns the number of outputs at the last layer of the network; or
   ## the number of inputs to the network if there are no layers.
   if self.layers.len < 1:
      return self.inputs.len
   else:
      return self.layers[self.layers.high].values.len

proc add_linear_layer*(self: SimpleNetwork, outputs: int) =
   self.layers.add(make_linear_layer(self.last_output_count, outputs))

proc add_tanh_layer*(self: SimpleNetwork, outputs: int) =
   self.layers.add(make_tanh_layer(self.last_output_count, outputs))

proc add_gru_layer*(self: SimpleNetwork, outputs: int) =
   self.layers.add(make_gru_layer(self.last_output_count, outputs))

