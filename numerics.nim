
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

