#pragma once

typedef struct _tensor_float_t tensor_float_t;

/* calculates mean squared errors for the loss function; ex. for showing
 * total loss */
float tensor_float_mse(tensor_float_t* tmp,
      tensor_float_t* results,
      tensor_float_t* goals);

/* calculates mean squared errors for the output layer. this is the
 * derivative of the loss function for each output neuron. */
void tensor_float_mse_derivs(tensor_float_t* dest,
      tensor_float_t* results,
      tensor_float_t* goals);

