
#include "random/random.h"
#include "random/cmwc.h"

#include "tensor.h"
#include "linear-layer.h"
#include "layer.h"

#include <time.h>
#include <assert.h>

int main(int argc, const char** argvs) {
   random_t* randomizer = random_cmwc_new((uint32_t)time(0));
   assert(randomizer);

   tensor_float_t* input = tensor_float_flat_new(3);
   assert(input);

   layer_t* layer = (layer_t*)linear_layer_new(3, 3);
   assert(layer);

   layer_t* layer2 = (layer_t*)linear_layer_new(3, 3);
   assert(layer2);

   layer_randomize_weights(layer, randomizer);
   layer_randomize_weights(layer2, randomizer);

   layer_forward(layer, input);
   layer_forward(layer2, layer->values);

   layer_free(layer);
   layer_free(layer2);

   random_free(randomizer);
}

