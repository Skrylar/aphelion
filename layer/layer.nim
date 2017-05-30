
import ../tensor

type
  Layer* = ref object {.inheritable.}
      values*, weights*: Tensor
      input_count*, value_count*, private_weight_count*: int

