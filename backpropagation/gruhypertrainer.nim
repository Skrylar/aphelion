
import ../random/cmwc
import ../criterion/criterion
import ../criterion/meansquarederrors
import ../simplenetwork
import ../tensor
import ../backpropagator
import ../layer/layer
import ../layer/gru

type
  ## An implementation of a self-training AI engine. It attempts to learn
  ## an optimal update function based on the network it has been asked to
  ## train, possibly learning multi-step update rules. Skrylar's
  ## implementation uses Gated Recurrent Units instead of
  ## Long Short-Term Memory units, as they have been empirically demonstrated
  ## to have equivalent performance while requiring substantially less
  ## computation time.
  ##
  ## Andrychowicz, Marcin et al. “Learning to Learn by Gradient Descent by Gradient Descent.” (2016): n. pag. Web. 4 Jan. 2017.
  GruHypertrainer* = ref object of Backpropagator
    hypertrainer*: SimpleNetwork
    hidden*: seq[seq[tuple[a, b: float64]]]
    hidden_private*: seq[seq[tuple[a, b: float64]]]
    update_cache*: Tensor
    backprop*: Backpropagator
    inputs*: Tensor
    goals*: Tensor
    mse*: Criterion

{.this:self.}

proc randomize_weights*(self: GruHypertrainer) =
  var randomizer = Cmwc()
  # TODO seed randomizer off system time
  randomizer.seed(1337)
  hypertrainer.randomize_weights randomizer

method init*(self: GruHypertrainer; net: SimpleNetwork) =
  ## Initializes the hyper-trainer to run on the supplied network.
  init_backpropagator(self, net)

  inputs = make_tensor(1)
  goals = make_tensor(1)

  update_cache = make_tensor(net.most_weights)

  # special training network
  hypertrainer = make_simple_network(inputs)
  hypertrainer.add_gru_layer(1)
  hypertrainer.add_tanh_layer(1)
  hypertrainer.add_linear_layer(1)
  hypertrainer.add_gru_layer(1)
  hypertrainer.add_tanh_layer(1)
  hypertrainer.add_linear_layer(1)

  # now randomize the net
  hypertrainer.auto_scratch_tensors()
  self.randomize_weights()

  backprop = Backpropagator()
  backprop.init hypertrainer
  mse = MseCriterion()

  # create hidden value map
  newseq(hidden, total_public_errors.len)
  for c in 0..(total_public_errors.len-1):
    newseq(hidden[c], total_public_errors[c].len)
    for j in 0..(total_public_errors[c].len - 1):
      hidden[c][j] = (a: 0.0, b: 0.0)
  newseq(hidden_private, total_private_errors.len)
  for c in 0..(total_private_errors.len-1):
    newseq(hidden_private[c], total_private_errors[c].len)
    for j in 0..(total_private_errors[c].len-1):
      hidden_private[c][j] = (a: 0.0, b: 0.0)

proc propagate*(self: GruHypertrainer; supervised: SimpleNetwork) =
  ## Makes corrections to a neural network based on each neuron's
  ## contribution to the network's error, and an internal neural
  ## network's proposed changes.

  for i in 0..hidden.high:
    # find updates for public neurons
    for j in 0..hidden[i].high:
      # unbox coordinate cells to our neural network
      let cells = hidden[i][j]
      hypertrainer.layers[0].GruLayer.hidden[0] = cells.a
      hypertrainer.layers[3].GruLayer.hidden[0] = cells.b
      # calculate adjustment for the neuron at this coordinate
      inputs[0] = total_public_errors[i][j]
      hypertrainer.forward()
      update_cache[j] = hypertrainer.output.values[0]
      # box new coordinate cells
      let new_cells = (a: hypertrainer.layers[0].GruLayer.hidden[0],
        b: hypertrainer.layers[3].GruLayer.hidden[0])
      hidden[i][j] = new_cells
    # push update cache to the layer
    supervised.layers[i].propagate(update_cache)
    # find updates for private neurons
    for j in 0..hidden_private[i].high:
      # unbox coordinate cells to our neural network
      let cells = hidden_private[i][j]
      hypertrainer.layers[0].GruLayer.hidden[0] = cells.a
      hypertrainer.layers[3].GruLayer.hidden[0] = cells.b
      # calculate adjustment for the neuron at this coordinate
      inputs[0] = total_public_errors[i][j]
      hypertrainer.forward()
      update_cache[j] = hypertrainer.output.values[0]
      # box new coordinate cells
      let new_cells = (a: hypertrainer.layers[0].GruLayer.hidden[0],
        b: hypertrainer.layers[3].GruLayer.hidden[0])
      hidden_private[i][j] = new_cells
    # push update cache to the layer
    supervised.layers[i].private_propagate(update_cache)

proc feedback*(self: GruHypertrainer; net: SimpleNetwork) =
  ## Analyzes the total output error of a trained neural network, and
  ## adjusts the internal optimizing network to make better corrections
  ## in future propagations.

  # calculate total loss of trained network
  var error = 0.0
  for x in 0..(total_public_errors[total_public_errors.high].len-1):
    error = error + total_public_errors[total_public_errors.high][x]
  for x in 0..(total_private_errors[total_private_errors.high].len-1):
    error = error + total_private_errors[total_private_errors.high][x]

  # update hypertrainer
  goals[0] = error
  backprop.clear_gradients()
  backprop.backward goals, mse, hypertrainer
  backprop.sgd 0.005, hypertrainer

