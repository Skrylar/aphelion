
import ../layer/layer
import ../tensor
import criterion

#_______________________________________________________________________
# Tensor maths
#_______________________________________________________________________

proc mse*(tmp: ScratchSet; results, goals: Tensor): float32 =
   ## calculates mean squared errors for the loss function; ex. for showing
   ## total loss
   assert tmp[0] != nil
   assert results != nil
   assert goals != nil

   set(tmp[0], 0)
   set_sub(tmp[0], results, goals)
   mul(tmp[0], tmp[0])
   mul(tmp[0], 0.5)

   result = hsum(tmp[0])

proc mse_derivs*(dest, results, goals: Tensor) =
   ## calculates mean squared errors for the output layer. this is the
   ## derivative of the loss function for each output neuron.
   assert dest != nil
   assert results != nil
   assert goals != nil

   set(dest, 0)
   set_sub(dest, results, goals)

#_______________________________________________________________________
# Criterion object
#_______________________________________________________________________

type
   MseCriterion* = ref object of Criterion
      ## Uses mean squared errors to determine the error from a final
      ## output layer and the correct labels.

method loss*(self: MseCriterion; scratch: ScratchSet; value_layer: Layer, goals: Tensor): float32 =
   result = mse(scratch, value_layer.values, goals)

method derive*(self: MseCriterion, value_layer: Layer, goals, output_error: Tensor) =
   mse_derivs(output_error, value_layer.values, goals)

