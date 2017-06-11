
import ../layer/layer
import ../tensor
import criterion

#_______________________________________________________________________
# Tensor maths
#_______________________________________________________________________

proc mse*(tmp, results, goals: Tensor): float32 =
   ## calculates mean squared errors for the loss function; ex. for showing
   ## total loss
   assert tmp != nil
   assert results != nil
   assert goals != nil

   set_sub(tmp, results, goals)
   mul(tmp, tmp)
   mul(tmp, 0.5)

   result = hsum(tmp)

proc mse_derivs*(dest, results, goals: Tensor) =
   ## calculates mean squared errors for the output layer. this is the
   ## derivative of the loss function for each output neuron.
   assert dest != nil
   assert results != nil
   assert goals != nil

   set_sub(dest, results, goals)

#_______________________________________________________________________
# Criterion object
#_______________________________________________________________________

type
   MseCriterion* = ref object of Criterion
      ## Uses mean squared errors to determine the error from a final
      ## output layer and the correct labels.

method loss*(value_layer: Layer, goals: Tensor): float32 =
   result = mse(value_layer.scratch[0], value_layer.values, goals)

method derive*(value_layer: Layer, goals, output_error: Tensor) =
   mse_derivs(output_error, value_layer.values, goals)

