
#include "criterion/mean-square-errors.h"

#include "random/random.h"
#include "random/cmwc.h"

#include "tensor.h"
#include "layer/linear-layer.h"
#include "layer/layer.h"

#include <stdio.h>
#include <time.h>
#include <assert.h>

#pragma GCC diagnostic ignored "-Wunused-variable"
int main(int argc, const char** argvs) {
   random_t* randomizer = random_cmwc_new((uint32_t)time(0));
   assert(randomizer);

   tensor_float_t* error_cache = tensor_float_flat_new(3*3);
   tensor_float_t* error_cache2 = tensor_float_flat_new(3*3);

   // create the network
   tensor_float_t* input = tensor_float_flat_new(3);
   assert(input);

   tensor_float_t* goals = tensor_float_flat_new(3);
   assert(goals);

   tensor_float_set_at(goals, 0, 0.0);
   tensor_float_set_at(goals, 0, 1.0);
   tensor_float_set_at(goals, 0, 0.0);

   layer_t* layer = (layer_t*)linear_layer_new(3, 3);
   assert(layer);

   layer_t* layer2 = (layer_t*)linear_layer_new(3, 3);
   assert(layer2);

   // initialize
   layer_randomize_weights(layer, randomizer);
   layer_randomize_weights(layer2, randomizer);

   for (int i = 0; i < 5; i++) {
   // forward pass
   layer_forward(layer, input);
   layer_forward(layer2, layer->values);

   // calculate errors
   float loss = tensor_float_mse(error_cache, layer2->values, goals);
   printf("Loss %f\n", loss);

   // work backwards
   {
      // start from derivatives of the goal function
      tensor_float_mse_derivs(error_cache, layer2->values, goals);
      tensor_float_set(error_cache2, error_cache);

      // apply derivatives
      layer_gradient(layer2, layer->values, error_cache);

      // backpropagate
      layer_propagate(layer2, error_cache);

      // apply derivatives to private weights
      layer_private_gradient(layer2, layer->values, error_cache);

      // backpropagate private weights
      layer_private_propagate(layer2, error_cache);
   }
   }

   // clean up
   layer_free(layer);
   layer_free(layer2);

   tensor_float_free(goals);
   tensor_float_free(error_cache);
   tensor_float_free(error_cache2);

   random_free(randomizer);
}

