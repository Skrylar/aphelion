
import ../tensor

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

   set(dest, results)
   sub(dest, goals)

