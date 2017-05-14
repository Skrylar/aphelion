
#include "tensor.h"

#include <assert.h>

#define min(x,y) (x < y) ? x : y

/* calculates mean squared errors for the loss function; ex. for showing
 * total loss */
float tensor_float_mse(tensor_float_t* tmp,
      tensor_float_t* results,
      tensor_float_t* goals)
{
   assert(tmp);
   assert(results);
   assert(goals);

   tensor_float_set(tmp, results);
   tensor_float_sub(tmp, goals);
   tensor_float_mul(tmp, tmp);
   tensor_float_mul1(tmp, 0.5);

   float result = tensor_float_hsum1(tmp);
   return result;
}

/* calculates mean squared errors for the output layer. this is the
 * derivative of the loss function for each output neuron. */
void tensor_float_mse_derivs(tensor_float_t* dest,
      tensor_float_t* results,
      tensor_float_t* goals)
{
   assert(dest);
   assert(results);
   assert(goals);

   tensor_float_set(dest, results);
   tensor_float_sub(dest, goals);
}

