
// XXX we use abort, but should not; need a proper exception handling
// system to deal with this crap

#include "tensor.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define checkself if (!self) abort();

int tensor_float_copy(tensor_float_t* dest, tensor_float_t* source,
      int dest_offset, int source_offset, int length)
{
   memcpy(dest->values + dest_offset,
	 source->values + source_offset,
	 length);
   return 0;
}

/* gives you a tensor created from ostensibly contiguous memory */
tensor_float_t* tensor_float_flat_new(int length) {
   tensor_float_t* self = calloc(1, sizeof(tensor_float_t));
   self->length = length;
   if (!self) return 0;
   self->values = malloc(sizeof(float)*length);
   if (!self->values) goto no_buffer;
   return self;
   
no_buffer:
   free(self);
   return 0;
}

/* A very basic form of convolution, used to carry inputs through
 * weights and in to an output.
 *
 * This procedure effectively implements the transfer of outputs from a
 * previous neural network layer, to a succeeding layer, through a
 * connected weight mechanism.
 */
void tensor_float_spread(tensor_float_t* source,
      tensor_float_t* weights,
      tensor_float_t* destination)
{
   // TODO elide these during NDEBUG builds
   if ((source == 0) || (weights == 0) || (destination == 0)) abort();
   if (weights->length < (source->length * destination->length)) abort();

   for (int i = 0; i < destination->length; i++) {
      for (int j = 0; j < source->length; j++) {
	 destination->values[i] += source->values[j] *
	    weights->values[(i * destination->length) + j];
      }
   }
}

void tensor_float_set1(tensor_float_t* self, float operand) {
   checkself;
   for (int i = 0; i < self->length; i++) {
      self->values[i] = operand;
   }
}

void tensor_float_set(tensor_float_t* self, tensor_float_t* operand) {
   checkself;
   if (self->length != operand->length) abort();
   for (int i = 0; i < self->length; i++) {
      self->values[i] = operand->values[i];
   }
}

void tensor_float_mul1(tensor_float_t* self, float operand) {
   checkself;
   for (int i = 0; i < self->length; i++) {
      self->values[i] *= operand;
   }
}

void tensor_float_mul(tensor_float_t* self, tensor_float_t* operand) {
   checkself;
   if (self->length != operand->length) abort();;
   for (int i = 0; i < self->length; i++) {
      self->values[i] *= operand->values[i];
   }
}

void tensor_float_div1(tensor_float_t* self, float operand) {
   checkself;
   for (int i = 0; i < self->length; i++) {
      self->values[i] /= operand;
   }
}

void tensor_float_div(tensor_float_t* self, tensor_float_t* operand) {
   checkself;
   if (self->length != operand->length) abort();;
   for (int i = 0; i < self->length; i++) {
      self->values[i] /= operand->values[i];
   }
}

void tensor_float_add1(tensor_float_t* self, float operand) {
   checkself;
   for (int i = 0; i < self->length; i++) {
      self->values[i] += operand;
   }
}

void tensor_float_add(tensor_float_t* self, tensor_float_t* operand) {
   checkself;
   if (self->length != operand->length) abort();;
   for (int i = 0; i < self->length; i++) {
      self->values[i] += operand->values[i];
   }
}

void tensor_float_sub1(tensor_float_t* self, float operand) {
   checkself;
   for (int i = 0; i < self->length; i++) {
      self->values[i] -= operand;
   }
}

void tensor_float_sub(tensor_float_t* self, tensor_float_t* operand) {
   checkself;
   if (self->length != operand->length) abort();;
   for (int i = 0; i < self->length; i++) {
      self->values[i] -= operand->values[i];
   }
}

void tensor_float_max1(tensor_float_t* self, float operand) {
   checkself;
   for (int i = 0; i < self->length; i++) {
      self->values[i] = fmax(self->values[i], operand);
   }
}

void tensor_float_max(tensor_float_t* self, tensor_float_t* operand) {
   checkself;
   if (self->length != operand->length) abort();;
   for (int i = 0; i < self->length; i++) {
      self->values[i] = fmax(self->values[i], operand->values[i]);
   }
}

void tensor_float_min1(tensor_float_t* self, float operand) {
   checkself;
   for (int i = 0; i < self->length; i++) {
      self->values[i] = fmin(self->values[i], operand);
   }
}

void tensor_float_min(tensor_float_t* self, tensor_float_t* operand) {
   checkself;
   if (self->length != operand->length) abort();;
   for (int i = 0; i < self->length; i++) {
      self->values[i] = fmin(self->values[i], operand->values[i]);
   }
}

//__attribute__((optimize("unroll-loops")))
//__attribute__((optimize("fast-math")))
float tensor_float_hsum1(tensor_float_t* self) {
   checkself;
   float accum = 0.0;
   for (int i = 0; i < self->length; i++) {
      accum += self->values[i];
   }
   return accum;
}

void tensor_float_free(tensor_float_t* self) {
   if (!self) return;
   if (self->values) free(self->values);
   free(self);
}

