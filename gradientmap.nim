
import tensor

type
  GradientMap* = seq[Tensor]

proc average*(self: var GradientMap, count: int) =
   ## Divides all numbers in the gradient map by a given count, which
   ## effectively averages across count samples. You must construct the
   ## gradient map by adding errors to it, instead of setting them, or
   ## else this will only divide the result of a single sample--which is
   ## not particularly helpful.
   if count < 2: return
   for i in 0..self.high:
      self[i].divide(float32(count))

proc clear*(self: var GradientMap) =
   ## Sets all numbers within the gradient map to zero.
   for i in 0..self.high:
      self[i].set(0.0)

