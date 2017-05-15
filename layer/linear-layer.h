#pragma once

#include "layer/layer.h"

typedef struct _linear_layer_t {
   layer_t super;

   tensor_float_t* private_weights;
   tensor_float_t* biases;
} linear_layer_t;

linear_layer_t* linear_layer_new(int inputs, int outputs);

