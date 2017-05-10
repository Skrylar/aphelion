
#include "random/random.h"
#include "random/cmwc.h"

#include "linear-layer.h"
#include "layer.h"

#include <time.h>

int main(int argc, const char** argvs) {
   random_t* randomizer = random_cmwc_new((uint32_t)time(0));

   layer_t* layer = (layer_t*)linear_layer_new(3, 3);
   layer_randomize_weights(layer, randomizer);

   layer_free(layer);
   random_free(randomizer);
}

