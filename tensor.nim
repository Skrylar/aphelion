
import math

type
  Tensor* = ref object
    data: seq[float32]

proc make_tensor*(size: int): Tensor =
  result = Tensor()
  newSeq(result.data, size)

# XXX i don't know what nim's equivalent to memcpy is
# XXX i don't know if we can disable bounds checking on sequences (or
# what voodoo is done to perform that.) this is performance sensitive
# code, and we especially want 

proc copy*(dest, source: Tensor) =
  dest.data.setLen(source.data.len)
  var hi = min(dest.data.high, source.data.high)
  for i in 0..hi:
    dest.data[i] = source.data[i]

proc copy*(dest, source: Tensor, dest_offset, source_offset, length: int) =
  # TODO sanity check those parameters
  var x = source_offset
  for i in dest_offset..(dest_offset+length):
    dest.data[i] = source.data[x]
    inc(x)

# XXX make destination the first param on these for consistency
proc spread*(source, weights, destination: Tensor) =
   ## A very basic form of convolution, used to carry inputs through
   ## weights and in to an output.
   ##
   ## This procedure effectively implements the transfer of outputs from a
   ## previous neural network layer, to a succeeding layer, through a
   ## connected weight mechanism.

   assert(source      != nil);
   assert(weights     != nil);
   assert(destination != nil);
   assert(weights.data.len >= (source.data.len * destination.data.len))

   for i in 0..destination.data.high:
      for j in 0..destination.data.high:
         destination.data[i] += source.data[j] *
            weights.data[(i * destination.data.len) + j]

proc set_spread*(source, weights, destination: Tensor) =
   assert(source      != nil);
   assert(weights     != nil);
   assert(destination != nil);

   assert(weights.data.len >= (source.data.len * destination.data.len))

   for i in 0..destination.data.high:
      for j in 0..destination.data.high:
         destination.data[i] = source.data[j] *
            weights.data[(i * destination.data.len) + j]

proc set*(self: Tensor, operand: float32) =
   for i in 0..self.data.high:
      self.data[i] = operand

proc set*(self, other: Tensor, count: int) =
   let x = min(self.data.high, other.data.high)
   for i in 0..x:
      self.data[i] = other.data[i]

proc set*(self, other: Tensor) =
   let x = min(self.data.high, other.data.high)
   self.set(other, x)

template set_at*(self: Tensor, pos: int, value: float) =
  self.data[pos] = value

template `[]=`*(self: Tensor, pos: int, value: float) =
   set_at(self, pos, value)

template get_at*(self: Tensor, pos: int): float =
  self.data[pos]

template `[]`*(self: Tensor, pos: int): float =
   get_at(self, pos)

proc max*(self: Tensor, operand: float32) =
   assert(self != nil)
   for i in 0..self.data.high:
      self.data[i] = max(self.data[i], operand)

proc max*(self, other: Tensor) =
   let x = min(self.data.high, other.data.high)
   for i in 0..x:
      self.data[i] = max(self.data[i], other.data[i])

proc min*(self: Tensor, operand: float32) =
   assert(self != nil)
   for i in 0..self.data.high:
      self.data[i] = min(self.data[i], operand)

proc min*(self, other: Tensor) =
   let x = min(self.data.high, other.data.high)
   for i in 0..x:
      self.data[i] = min(self.data[i], other.data[i])

proc exp*(self: Tensor) =
   for i in 0..self.data.high:
      self.data[i] = exp(self.data[i])

template defop(name, set_name, op: untyped) =
   proc name*(self: Tensor, operand: float32) =
      for i in 0..self.data.high:
         self.data[i] = op(self.data[i], operand)

   proc name*(dest, operand: Tensor) =
      let x = min(dest.data.high, operand.data.high)
      for i in 0..x:
         dest.data[i] = op(dest.data[i], operand.data[i])

   proc set_name*(dest, operand1, operand2: Tensor) =
      let x = min(operand1.data.high, operand2.data.high)
      for i in 0..x:
         dest.data[i] = op(operand1.data[i], operand2.data[i])

   proc name*(dest, operand: Tensor, count: int) =
      for i in 0..(count-1):
         dest.data[i] = op(dest.data[i], operand.data[i])

   proc name*(dest, operand: Tensor,
      dest_offset, source_offset, count: int) =
         var at = source_offset
         for i in dest_offset..((source_offset+count)-1):
            dest.data[i] = op(dest.data[i], operand.data[at])
            inc(at)

defop(`add`, `set_add`, `+`)
defop(`mul`, `set_mul`, `*`)
defop(`divide`, `set_divide`, `/`)
defop(`sub`, `set_sub`, `-`)

proc tanh*(self: Tensor, count: int) =
   ## hyperbolic tangent
   ## https://en.wikipedia.org/wiki/Hyperbolic_function */
   assert(self != nil)

   # pre-calculate E
   for i in 0..(count-1):
      self.data[i] = exp(-(2 * self.data[i]));

   # now calculate tangent
   for i in 0..(count-1):
      self.data[i] = (1 - self.data[i]) / (1 + self.data[i])

proc tanh*(self: Tensor) =
   self.tanh(self.data.len)

proc tanh_deriv*(self: Tensor) =
   ## hyperbolic tangent's derivative
   ## derived with SAGE
   assert(self != nil);

   # turn all of our own values in to e
   self.mul(-2)
   self.exp

   # 2 * (1 - e) * e / (1 + pow(e, 2) + 2 * e / (1 + e));

   # this one is going to be "fun" to write an AVX version of
   for i in 0..self.data.high:
      self.data[i] = 2 * (1 - self.data[i]) * self.data[i] / (1 + pow(self.data[i], 2) + 2 * self.data[i] / (1 + self.data[i]))

proc hsum*(self: Tensor): float32 =
   assert(self != nil)
   result = 0.0
   for i in 0..self.data.high:
      result = result + self.data[i]

proc len*(self: Tensor): int =
  return self.data.len

proc weight_sum*(dest, weights: Tensor, weight_count, value_count: int) =
   ## takes a tensor of weights and sums those in to a smaller operand; so
   ## if 'self' is 10 weights and 5 for each neuron, the operand will have
   ## a length of two, and be filled with the sums of those weights. this
   ## lets us calculate the amount of "fault" a given neuron is at.

   assert(dest != nil)
   assert(weights != nil)
   assert((weight_count %% value_count) == 0)

   var stripe_length = weight_count /% valuecount
   var stripes = weight_count /% stripe_length
   var at = 0

   for s in 0..(stripes - 1):
      for i in 0..(stripe_length - 1):
         dest.data[s] = dest.data[s] + weights.data[at]
         inc(at)

proc weight_sum*(dest, weights: Tensor) =
   weight_sum(dest, weights, dest.len, weights.len)

when isMainModule:
   import unittest

   suite "Backpropagation primitives":
      test "Calculating sums of weights going backward":
         # This operation is done during backpropagation; you take the
         # weights of neurons in a future layer, work out the weights of
         # a previous neuron, and distribute the error backwards
         # proportionately.
         var a = make_tensor(4)
         var c = make_tensor(2)

         a[0] = 0.5
         a[1] = 0.25

         a[2] = 0.25
         a[3] = 0.1

         weight_sum(c, a, 4, 2)
         check(abs(c[0] - 0.75) < 0.1)
         check(abs(c[1] - 0.35) < 0.1)

