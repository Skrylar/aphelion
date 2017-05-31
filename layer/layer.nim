
import ../tensor

type
  Layer* = ref object {.inheritable.}
      values*, weights*: Tensor
      input_count*, value_count*, private_weight_count*: int

      # scratch space for calculations
      scratch*: array[0..2, Tensor]
