
import ../tensor
import ../random/random

type
  ScratchSet* = array[0..4, Tensor]

  Layer* = ref object {.inheritable.}
      values*, weights*: Tensor
      input_count*, internal_weights*: int

method forward*(self: Layer; inputs: Tensor; scratch: ScratchSet) {.base.} =
   discard

method gradient*(self: Layer; inputs, deltas, total: Tensor; scratch: ScratchSet) {.base.} =
   discard

method propagate*(self: Layer; updates: Tensor; scratch: ScratchSet) {.base.} =
   self.weights.sub(updates)

method randomize_weights*(self: Layer, rng: Random) {.base.} =
   for i in 0..(self.weights.len - 1):
      self.weights.set_at(i, rng.next_float);

