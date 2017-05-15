
#include "random/random.h"
#include "linear-layer.h"
#include "tensor.h"

#include <stdlib.h>

static int open(linear_layer_t* self) {
   self->super.values = tensor_float_flat_new(self->super.value_count);
   if (!self->super.values) goto no_values;

   self->super.weights = tensor_float_flat_new((self->super.value_count) * self->super.input_count);
   if (!self->super.weights) goto no_weights;

   self->biases = tensor_float_flat_new(self->super.value_count);
   if (!self->biases) goto no_biases;
   tensor_float_set1(self->biases, 0);

   self->private_weights = tensor_float_flat_new(self->super.value_count);
   if (!self->private_weights) goto no_private_weights;

   return 0;

no_private_weights:
   tensor_float_free(self->biases);
no_biases:
   tensor_float_free(self->super.weights);
no_weights:
   tensor_float_free(self->super.values);
no_values:
   return 1;
}

static void close(linear_layer_t* self) {
   tensor_float_free(self->super.values);
   tensor_float_free(self->super.weights);
   tensor_float_free(self->private_weights);
   tensor_float_free(self->biases);
   free(self);
}

static void forward(linear_layer_t* self, tensor_float_t* inputs) {
   tensor_float_set1(self->super.values, 0);
   tensor_float_spread(inputs, self->super.weights, self->super.values);
   tensor_float_div1(self->super.values, inputs->length);
   tensor_float_add(self->super.values, self->biases);
}

static void gradient(linear_layer_t* self,
		tensor_float_t* inputs,
		tensor_float_t* deltas)
{
   // get the big stuff
   tensor_float_mul(deltas, inputs);
}

static void private_gradient(linear_layer_t* self,
		tensor_float_t* inputs,
		tensor_float_t* deltas)
{
   // fix up the private deltas
   tensor_float_set(deltas, self->biases);
}

static void propagate(linear_layer_t* self, tensor_float_t* updates) {
   tensor_float_sub(self->super.weights, updates);
}

static void private_propagate(linear_layer_t* self, tensor_float_t* updates) {
   tensor_float_sub(self->private_weights, updates);
}

static void randomize_weights(linear_layer_t* self, random_t* rng) {
   for (int i = 0; i < self->super.weights->length; i++) {
      tensor_float_set_at(self->super.weights, i, random_next_float(rng));
   }

   for (int i = 0; i < self->private_weights->length; i++) {
      tensor_float_set_at(self->private_weights, i, random_next_float(rng));
   }

   for (int i = 0; i < self->biases->length; i++) {
      tensor_float_set_at(self->biases, i, random_next_float(rng));
   }
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wincompatible-pointer-types"
static layer_class_t linear_layer_class = {
   &close,
   &forward,
   &gradient,
   &private_gradient,
   &propagate,
   &private_propagate,
   &randomize_weights
};
#pragma GCC diagnostic pop

linear_layer_t* linear_layer_new(int inputs, int outputs) {
   linear_layer_t* self = calloc(1, sizeof(linear_layer_t));
   if (!self) goto no_self;

   self->super.cls = &linear_layer_class;
   self->super.input_count = inputs;
   self->super.value_count = outputs;
   /* one private weight for each layer */
   self->super.private_weight_count = inputs;

   if (open(self) != 0) goto fucked;

   return self;

fucked:
   free(self);
no_self:
   return 0;
}

