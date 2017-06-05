
import ../tensor
import ../random/random

type
  Layer* = ref object {.inheritable.}
      values*, weights*: Tensor
      input_count*, value_count*, private_weight_count*: int

      # scratch space for calculations
      scratch*: array[0..2, Tensor]

method open*(self: Layer) {.base.} =
   discard

method close*(self: Layer) {.base.} =
   discard

method forward*(self: Layer, inputs: Tensor) {.base.} =
   discard

method gradient*(self: Layer, inputs, deltas, total: Tensor) {.base.} =
   discard

method private_gradient*(self: Layer, inputs, deltas, total: Tensor) {.base.} =
   discard
 
method propagate*(self: Layer, updates: Tensor) {.base.} =
   discard

method private_propagate*(self: Layer, updates: Tensor) {.base.} =
   discard

method randomize_weights*(self: Layer, rng: Random) {.base.} =
   discard

