#pragma once

#include "layer/layer.h"

typedef struct _tanh_layer_t {
   layer_t super;
} tanh_layer_t;

tanh_layer_t* tanh_layer_new(int inputs, int outputs);

