
#include "random/random.h"
#include "tanh-layer.h"
#include "tensor.h"

#include <stdlib.h>

static int open(tanh_layer_t* self) {
   self->super.values = tensor_float_flat_new(self->super.value_count);
   if (!self->super.values) goto no_values;

   self->super.weights = tensor_float_flat_new((self->super.value_count) * self->super.input_count);
   if (!self->super.weights) goto no_weights;

   return 0;

no_weights:
   tensor_float_free(self->super.values);
no_values:
   return 1;
}

static void close(tanh_layer_t* self) {
   tensor_float_free(self->super.values);
   tensor_float_free(self->super.weights);
   free(self);
}

static void forward(tanh_layer_t* self, tensor_float_t* inputs) {
   tensor_float_set1(self->super.values, 0);
   tensor_float_spread(inputs, self->super.weights, self->super.values);
   tensor_float_tanh(self->super.values);
}

static void gradient(tanh_layer_t* self,
		tensor_float_t* inputs,
		tensor_float_t* deltas)
{
   // get the big stuff
   tensor_float_mul(deltas, inputs);
   tensor_float_tanh_deriv(deltas);
}

static void private_gradient(tanh_layer_t* self,
		tensor_float_t* inputs,
		tensor_float_t* deltas)
{
}

static void propagate(tanh_layer_t* self, tensor_float_t* updates) {
   tensor_float_sub(self->super.weights, updates);
}

static void private_propagate(tanh_layer_t* self, tensor_float_t* updates) { }

static void randomize_weights(tanh_layer_t* self, random_t* rng) {
   for (int i = 0; i < self->super.weights->length; i++) {
      tensor_float_set_at(self->super.weights, i, random_next_float(rng));
   }
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wincompatible-pointer-types"
static layer_class_t tanh_layer_class = {
   &close,
   &forward,
   &gradient,
   &private_gradient,
   &propagate,
   &private_propagate,
   &randomize_weights
};
#pragma GCC diagnostic pop

tanh_layer_t* tanh_layer_new(int inputs, int outputs) {
   tanh_layer_t* self = calloc(1, sizeof(tanh_layer_t));
   if (!self) goto no_self;

   self->super.cls = &tanh_layer_class;
   self->super.input_count = inputs;
   self->super.value_count = outputs;
   /* one private weight for each layer */
   self->super.private_weight_count = 0;

   if (open(self) != 0) goto fucked;

   return self;

fucked:
   free(self);
no_self:
   return 0;
}

