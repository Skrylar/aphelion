#pragma once

#include "layer/layer.h"

typedef struct _gru_layer_t {
   layer_t super;

   tensor_float_t* hidden;
   tensor_float_t* reset_weights;
   tensor_float_t* reset_hidden_weights;
   tensor_float_t* update_weights;
   tensor_float_t* update_hidden_weights;
   tensor_float_t* hhat_weights;
} gru_layer_t;

gru_layer_t* gru_layer_new(int inputs, int outputs);

