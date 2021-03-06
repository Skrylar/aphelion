#pragma once

#include <assert.h>

/* opaque references */
typedef struct _random_t random_t;
typedef struct _tensor_float_t tensor_float_t;

/* virtual functions */
typedef void (*layer_close_f)(void* self);
typedef void (*layer_forward_f)(void* self, tensor_float_t* inputs);
typedef void (*layer_gradient_f)(void* self, tensor_float_t* inputs, tensor_float_t* deltas);
typedef void (*layer_propagate_f)(void* self, tensor_float_t* changes);
typedef void (*layer_randomize_weights_f)(void* self, random_t* changes);

typedef struct _layer_class_t {
   layer_close_f close;
   layer_forward_f forward;
   layer_gradient_f gradient;
   layer_gradient_f private_gradient;
   layer_propagate_f propagate;
   layer_propagate_f private_propagate;
   layer_randomize_weights_f randomize_weights;
} layer_class_t;

typedef struct _layer_t {
   /* class information for a particular layer type */
   layer_class_t* cls;
   /* how many neurons exist in this layer */
   int value_count;
   /* how many private weights per value are in this layer */
   int private_weight_count;
   /* how many neurons exist in the previous layer? */
   int input_count;

   tensor_float_t* values;
   tensor_float_t* weights;

   tensor_float_t* scratch[3];
} layer_t;

__attribute__((always_inline))
inline void layer_randomize_weights(layer_t* self, random_t* rng) {
   assert(self);
   self->cls->randomize_weights(self, rng);
}

__attribute__((always_inline))
inline void layer_forward(layer_t* self, tensor_float_t* inputs) {
   assert(self);
   assert(inputs);
   self->cls->forward(self, inputs);
}

__attribute__((always_inline))
inline void layer_gradient(layer_t* self, tensor_float_t* inputs, tensor_float_t* deltas) {
   assert(self);
   assert(inputs);
   assert(deltas);
   self->cls->gradient(self, inputs, deltas);
}

__attribute__((always_inline))
inline void layer_private_gradient(layer_t* self, tensor_float_t* inputs, tensor_float_t* deltas) {
   assert(self);
   assert(inputs);
   assert(deltas);
   self->cls->private_gradient(self, inputs, deltas);
}

__attribute__((always_inline))
inline void layer_propagate(layer_t* self, tensor_float_t* changes) {
   assert(self);
   assert(changes);
   self->cls->propagate(self, changes);
}

__attribute__((always_inline))
inline void layer_private_propagate(layer_t* self, tensor_float_t* changes) {
   assert(self);
   assert(changes);
   self->cls->private_propagate(self, changes);
}

__attribute__((always_inline))
inline void layer_free(layer_t* self) {
   assert(self);
   self->cls->close(self);
}

