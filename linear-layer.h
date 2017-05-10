#pragma once

#include "layer.h"

typedef struct _linear_layer_t {
   layer_t super;

   tensor_float_t* values;
   tensor_float_t* weights;
   tensor_float_t* private_weights;
   tensor_float_t* biases;
} linear_layer_t;

linear_layer_t* linear_layer_new(int inputs, int outputs);

