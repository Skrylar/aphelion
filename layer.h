#pragma once

#include "tensor.h"

typedef int (*layer_close_f)(void* self);
typedef int (*layer_forward_f)(void* self, tensor_float_t* inputs);
typedef int (*layer_gradient_f)(void* self, tensor_float_t* inputs, tensor_float_t* deltas);
typedef int (*layer_propagate_f)(void* self, tensor_float_t* changes);

typedef struct _layer_class_t {
   layer_close_f* close;
   layer_forward_f* forward;
   layer_gradient_f* gradient;
   layer_gradient_f* private_gradient;
   layer_propagate_f* propagate;
   layer_propagate_f* private_propagate;
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
} layer_t;

