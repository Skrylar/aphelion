
import math

when isMainModule:
  import unittest

proc additivelySmooth*[T:SomeReal](hits, trials, pseudocount: T): T =
  ## Performs laplace smoothing on a given number of hits, total trial
  ## count, and the pseudocount of trials to be virtually added.
  return (hits + pseudocount) / ((pseudocount * trials) + trials)

when isMainModule:
  test "Additive smoothing":
    var v = additivelySmooth(2.0, 2.0, 1.0)
    var j = additivelySmooth(0.0, 2.0, 1.0)

    check v < 1
    check v > 0
    check j > 0

proc bayesianAverage*[T:SomeReal](score, count, mean, pseudocount: T): T =
  ## Pads a score and count of real events, with a pseudocount of
  ## events equal to mean. Useful for being a hack of a data
  ## scientist, or adding a number of fake votes for ranking systems
  ## to prevent a small number of good ratings from seeming more
  ## important than a large number of average ratings.
  return ((mean * pseudocount) + score) / (pseudocount + count)

when isMainModule:
  test "Bayesian average":
    # Book 1 has one rating of 5, and book 2 has fifty ratings with an average of 4.5. https://fulmicoton.com/posts/bayesian_rating/
    var book1 = bayesianAverage(5.0, 1.0, 3.0, 5.0)
    var book2 = bayesianAverage((4.5 * 50), 50.0, 3.0, 5.0)
    check book2 > book1
