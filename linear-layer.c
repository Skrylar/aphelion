
#include "layer.h"
#include "tensor.h"

#include <stdlib.h>

typedef struct _linear_layer_t {
   layer_t super;

   tensor_float_t* values;
   tensor_float_t* weights;
   tensor_float_t* private_weights;
   tensor_float_t* biases;
} linear_layer_t;

static int open(linear_layer_t* self) {
   self->values = tensor_float_flat_new(self->super.value_count);
   if (!self->values) goto no_values;

   self->weights = tensor_float_flat_new((self->super.value_count) * self->super.input_count);
   if (!self->weights) goto no_weights;

   self->biases = tensor_float_flat_new(self->super.value_count);
   if (!self->biases) goto no_biases;

   return 0;

no_biases:
   tensor_float_free(self->weights);
no_weights:
   tensor_float_free(self->values);
no_values:
   return 1;
}

static void close(linear_layer_t* self) {
   tensor_float_free(self->values);
   tensor_float_free(self->weights);
   tensor_float_free(self->biases);
}

static void forward(linear_layer_t* self, tensor_float_t* inputs) {
   tensor_float_set(self->values, 0);
   tensor_float_spread(inputs, self->weights, self->values);
   tensor_float_div1(self->values, inputs->length);
   tensor_float_add(self->values, self->biases);
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
   tensor_float_sub(self->weights, updates);
}

static void private_propagate(linear_layer_t* self, tensor_float_t* updates) {
   tensor_float_sub(self->private_weights, updates);

}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wincompatible-pointer-types"
static layer_class_t linear_layer_class = {
   &close,
   &forward,
   &gradient,
   &private_gradient,
   &propagate,
   &private_propagate
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

