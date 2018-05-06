import math, fenv

when isMainModule:
  import unittest

proc within*[T: float | float32 | float64](value, target, epsilon: T): bool =
  ## Checks if a real number is within a given threshold of a target
  ## value.
  return abs(value - target) < epsilon

when isMainModule:
  test "Within":
    check(within(5.0, 5.5, 1.0) == true)
    check(within(5.0, 4.5, 1.0) == true)
    check(within(5.0, 5.5, 0.1) == false)

proc rescale*[T: float | float32 | float64](value, old_min, old_max, minn, maxx: T): T =
  ## Changes a real number from one arbitrary bounded scale to a new one.
  let percentage = (value - old_min) / (old_max - old_min)
  return (percentage * (maxx - minn)) + minn

when isMainModule:
  test "Rescale":
    check(within(rescale(0.5, 0.0, 1.0, 5.0, 10.0), 7.5, 0.1) == true)

proc sum*[T](input: openarray[T]): T =
  ## Computes the sum of a given array.
  result = 0.T
  for value in input:
    result = result + value

when isMainModule:
  test "Sum":
    let integers = [1, 2, 4, 8]
    let reals = [1.0, 2.0, 4.0, 8.0]
    check sum(integers) == 15
    check within(sum(reals), 15.0, 0.1) == true

proc average*[T](input: openarray[T]): T =
  ## Computes the sum of a given array.
  result = 0.T
  for value in input:
    result = result + value
  result = (result / input.len.T).T

template mean*[T](input: openarray[T]): T =
  average(input)

when isMainModule:
  test "Average":
    let reals = [1.0, 2.0, 4.0, 8.0]
    check within(average(reals), 3.75, 0.1) == true

proc cosineSimilarity*[T:SomeReal](a, b: openarray[T]): T =
  var dot = 0.T
  var leftMagnitude = 0.T
  var rightMagnitude = 0.T

  for i in 0..<min(a.len, b.len):
    dot += a[i] * b[i]
    leftMagnitude += a[i] * a[i]
    rightMagnitude += b[i] * b[i]

  return dot / ((leftMagnitude.sqrt * rightMagnitude.sqrt)).max(epsilon(T))

when isMainModule:
  test "Cosine similarity":
    var a = [5.0, 5.0, 5.0]
    var b = [0.0, 0.0, 0.0]
    check within(cosineSimilarity(a, a), 1.0, 0.1)
    check within(cosineSimilarity(b, b), 0.0, 0.1)