
import ../layer/layer
import ../tensor

type
   Criterion* = ref object {.inheritable.}
      ## Uses mean squared errors to determine the error from a final
      ## output layer and the correct labels.

method loss*(value_layer: Layer, goals: Tensor): float32 {.base.} =
   discard

method derive*(value_layer: Layer, goals, output_error: Tensor) {.base.} =
   discard

