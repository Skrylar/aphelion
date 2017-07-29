
import math
import ../layer/layer
import ../simplenetwork
import ../tensor
import ../backpropagator

type
  ## ADAM appears to be the second prevailing training system according
  ## to 2016's research, with L-BFGS coming out ahead (while requiring
  ## significantly more memory.)
  ##
  ## Kingma, Diederik, and Jimmy Ba. “Adam: A Method for Stochastic
  ## Optimization.” arXiv (2015): n. pag. Web. 22 Dec. 2016.
  AdamBackpropagator = ref object of Backpropagator
    alpha*: float64 ## step size.
    beta1*: float64 ## exponential decay rate 1.
    beta2*: float64 ## exponential decay rate 2.
    epsilon*: float64

    t*: int ## current iteration of the backpropagator. you want to set this to one when starting over with a new training epoch.
    moment_pa*, moment_xa*, moment_pb, moment_xb*: seq[Tensor] ## Moment vectors used for parameter adjustments. P is for public weights, X is for private ones. A an B are the two moment pairs.
    update_cache_a, update_cache_b*: Tensor ## Used to accumulate change values before passing them off to neuron layers.

method init*(self: AdamBackpropagator, net: SimpleNetwork) =
  # these are taken from the whitepaper; you may desire to change them
  self.t       = 1
  self.alpha   = 0.001
  self.beta1   = 0.9
  self.beta2   = 0.999
  self.epsilon = 0.00000001
  # create caches
  self.update_cache_a = make_tensor(net.most_weights)
  self.update_cache_b = make_tensor(net.most_weights)
  # create moment sequences
  newseq(self.moment_pa, net.layers.len)
  newseq(self.moment_pb, net.layers.len)
  newseq(self.moment_xa, net.layers.len)
  newseq(self.moment_xb, net.layers.len)
  # fill moment sequences with tensors
  for x in 0..net.layers.high:
    self.moment_pa[x] = make_tensor(net.layers[x].weights.len)
    self.moment_pb[x] = make_tensor(net.layers[x].weights.len)
    self.moment_xa[x] = make_tensor(net.layers[x].private_weight_count)
    self.moment_xb[x] = make_tensor(net.layers[x].private_weight_count)

{.this:self.}
  
proc propagate*(self: AdamBackpropagator, network: SimpleNetwork) =
  for i in 0..network.layers.high:
    template layer(): untyped = network.layers[i]
    template squared(x: untyped): untyped = x * x
    for j in 0..(total_public_errors[i].len-1):
      # update Mt
      var mt = (beta1 * moment_pa[i][j]) + ((1 - beta1) * total_public_errors[i][j])
      var mtx = (beta1 * moment_xa[i][j]) + ((1 - beta1) * total_private_errors[i][j])
      moment_pa[i][j] = mt
      moment_xa[i][j] = mtx

      # update Vt
      var vt = (beta1 * moment_pb[i][j]) + ((1 - beta2) * squared(total_public_errors[i][j]))
      var vtx = (beta1 * moment_xb[i][j]) + ((1 - beta2) * squared(total_private_errors[i][j]))
      moment_pb[i][j] = vt
      moment_xb[i][j] = vtx

      # hats
      var mhat = mt / (1 - pow(beta1, t.float64))
      var mhatx = mtx / (1 - pow(beta1, t.float64))
      var vhat = vt / (1 - pow(beta2, t.float64))
      var vhatx = vtx / (1 - pow(beta2, t.float64))
      inc(t)

      # store
      update_cache_a[j] = alpha * (mhat / (sqrt(vhat) + epsilon))
      update_cache_b[j] = alpha * (mhatx / (sqrt(vhatx) + epsilon))
    
    layer.propagate(update_cache_a)
    layer.private_propagate(update_cache_b)
