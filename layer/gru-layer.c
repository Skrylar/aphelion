
#include "random/random.h"
#include "gru-layer.h"
#include "tensor.h"

#include <stdlib.h>

static int open(gru_layer_t* self) {
   self->super.values = tensor_float_flat_new(self->super.value_count);
   if (!self->super.values) goto no_values;

   self->super.weights = tensor_float_flat_new((self->super.value_count) * self->super.input_count);
   if (!self->super.weights) goto no_weights;

   self->hidden = tensor_float_flat_new((self->super.value_count));
   if (!self->hidden) goto no_hidden;
   self->reset_weights = tensor_float_flat_new((self->super.value_count));
   if (!self->reset_weights) goto no_reset_weights;
   self->reset_hidden_weights = tensor_float_flat_new((self->super.value_count));
   if (!self->reset_hidden_weights) goto no_reset_hidden_weights;
   self->update_weights = tensor_float_flat_new((self->super.value_count));
   if (!self->update_weights) goto no_update_weights;
   self->update_hidden_weights = tensor_float_flat_new((self->super.value_count));
   if (!self->update_hidden_weights) goto no_update_hidden_weights;
   self->hhat_weights = tensor_float_flat_new((self->super.value_count));
   if (!self->hhat_weights) goto no_hhat_weights;

no_hhat_weights:
   tensor_float_free(self->update_hidden_weights);
no_update_hidden_weights:
   tensor_float_free(self->update_weights);
no_update_weights:
   tensor_float_free(self->reset_hidden_weights);
no_reset_hidden_weights:
   tensor_float_free(self->reset_weights);
no_reset_weights:
   tensor_float_free(self->hidden);
no_hidden:
   tensor_float_free(self->super.weights);
no_weights:
   tensor_float_free(self->super.values);
no_values:
   return 1;
}

static void close(gru_layer_t* self) {
   tensor_float_free(self->super.values);
   tensor_float_free(self->super.weights);

   tensor_float_free(self->hidden);
   tensor_float_free(self->reset_weights);
   tensor_float_free(self->reset_hidden_weights);
   tensor_float_free(self->update_weights);
   tensor_float_free(self->update_hidden_weights);
   tensor_float_free(self->hhat_weights);

   free(self);
}

static void forward(gru_layer_t* self, tensor_float_t* inputs) {
   tensor_float_set(self->hidden, self->super.values);

   tensor_float_set_spread(inputs, self->super.weights, self->super.values);

   assert(self->super.scratch[0]);
   assert(self->super.scratch[1]);
   assert(self->super.scratch[2]);

   /* calculate value of the reset neuron input */
   tensor_float_set_mul(self->super.scratch[0],
	 self->reset_weights,
	 self->super.values);

   tensor_float_set_mul(self->super.scratch[1],
	 self->reset_hidden_weights,
	 self->hidden);

   tensor_float_add_len(self->super.scratch[0], self->super.scratch[1],
	 self->hidden->length);
   tensor_float_tanh_len(self->super.scratch[0], self->hidden->length);

   /* calculate value of the update neuron input */
   tensor_float_set_mul(self->super.scratch[1],
	 self->update_weights,
	 self->super.values);

   tensor_float_set_mul(self->super.scratch[2],
	 self->update_hidden_weights,
	 self->hidden);

   tensor_float_add_len(self->super.scratch[1], self->super.scratch[2],
	 self->hidden->length);
   tensor_float_tanh(self->super.scratch[1]);

   // var hhat = Curves.Tanh(Values[i] + (Weights[x++] * (reset * Hidden[i])));
   tensor_float_set_mul(self->super.scratch[2], self->super.scratch[0], self->hidden);
   tensor_float_mul(self->super.scratch[2], self->hhat_weights);
   tensor_float_add(self->super.scratch[2], self->super.values);

   tensor_float_set1(self->super.scratch[0], 1.0);
   tensor_float_sub(self->super.scratch[0], self->super.scratch[1]);
   tensor_float_mul(self->super.scratch[2], self->super.scratch[0]);

   tensor_float_set_mul(self->super.values, self->super.scratch[1], self->hidden);
   tensor_float_add(self->super.values, self->super.scratch[2]);
}

static void gradient(gru_layer_t* self,
		tensor_float_t* inputs,
		tensor_float_t* deltas)
{
   // get the big stuff
   tensor_float_mul(deltas, inputs);
}

static void private_gradient(gru_layer_t* self,
		tensor_float_t* inputs,
		tensor_float_t* deltas)
{
   // TODO
}

static void propagate(gru_layer_t* self, tensor_float_t* updates) {
   tensor_float_sub(self->super.weights, updates);
}

static void private_propagate(gru_layer_t* self, tensor_float_t* updates) {
   // TODO
}

static void randomize_weights(gru_layer_t* self, random_t* rng) {
#define reee(x) for (int i = 0; i < x->length; i++) { tensor_float_set_at(x, i, random_next_float(rng)); }

   reee(self->super.weights);

   reee(self->reset_weights);
   reee(self->reset_hidden_weights);
   reee(self->update_weights);
   reee(self->update_hidden_weights);
   reee(self->hhat_weights);

#undef reee
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wincompatible-pointer-types"
static layer_class_t gru_layer_class = {
   &close,
   &forward,
   &gradient,
   &private_gradient,
   &propagate,
   &private_propagate,
   &randomize_weights
};
#pragma GCC diagnostic pop

gru_layer_t* gru_layer_new(int inputs, int outputs) {
   gru_layer_t* self = calloc(1, sizeof(gru_layer_t));
   if (!self) goto no_self;

   self->super.cls = &gru_layer_class;
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

