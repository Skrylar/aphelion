
// XXX we use abort, but should not; need a proper exception handling
// system to deal with this crap

#include "tensor.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#define min(x,y) ((x < y) ? x : y)

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
   assert(length > 0);

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
   assert(source);
   assert(weights);
   assert(destination);
   assert(weights->length >= (source->length * destination->length));

   for (int i = 0; i < destination->length; i++) {
      for (int j = 0; j < source->length; j++) {
	 destination->values[i] += source->values[j] *
	    weights->values[(i * destination->length) + j];
      }
   }
}

void tensor_float_set1(tensor_float_t* self, float operand) {
   assert(self);
   for (int i = 0; i < self->length; i++) {
      self->values[i] = operand;
   }
}

void tensor_float_set(tensor_float_t* self, tensor_float_t* operand) {
   assert(self);
   assert(operand);

   int len = min(self->length, operand->length);
   for (int i = 0; i < len; i++) {
      self->values[i] = operand->values[i];
   }
}

void tensor_float_mul1(tensor_float_t* self, float operand) {
   assert(self);
   for (int i = 0; i < self->length; i++) {
      self->values[i] *= operand;
   }
}

void tensor_float_mul(tensor_float_t* self, tensor_float_t* operand) {
   assert(self);
   assert(operand);
   if (self->length != operand->length) abort();;
   for (int i = 0; i < self->length; i++) {
      self->values[i] *= operand->values[i];
   }
}

void tensor_float_div1(tensor_float_t* self, float operand) {
   assert(self);
   for (int i = 0; i < self->length; i++) {
      self->values[i] /= operand;
   }
}

void tensor_float_div(tensor_float_t* self, tensor_float_t* operand) {
   assert(self);
   assert(operand);
   if (self->length != operand->length) abort();;
   for (int i = 0; i < self->length; i++) {
      self->values[i] /= operand->values[i];
   }
}

void tensor_float_add1(tensor_float_t* self, float operand) {
   assert(self);
   assert(operand);
   for (int i = 0; i < self->length; i++) {
      self->values[i] += operand;
   }
}

void tensor_float_add(tensor_float_t* self, tensor_float_t* operand) {
   assert(self);
   assert(operand);
   if (self->length != operand->length) abort();;
   for (int i = 0; i < self->length; i++) {
      self->values[i] += operand->values[i];
   }
}

void tensor_float_sub1(tensor_float_t* self, float operand) {
   assert(self);
   for (int i = 0; i < self->length; i++) {
      self->values[i] -= operand;
   }
}

void tensor_float_sub(tensor_float_t* self, tensor_float_t* operand) {
   assert(self);
   assert(operand);
   int len = min(self->length, operand->length);
   for (int i = 0; i < len; i++) {
      self->values[i] -= operand->values[i];
   }
}

void tensor_float_max1(tensor_float_t* self, float operand) {
   assert(self);
   for (int i = 0; i < self->length; i++) {
      self->values[i] = fmax(self->values[i], operand);
   }
}

__attribute__((nonnull))
void tensor_float_max(tensor_float_t* self, tensor_float_t* operand) {
   assert(self);
   assert(operand);
   int len = min(self->length, operand->length);
   for (int i = 0; i < len; i++) {
      self->values[i] = fmax(self->values[i], operand->values[i]);
   }
}

__attribute__((nonnull))
void tensor_float_min1(tensor_float_t* self, float operand) {
   assert(self);
   for (int i = 0; i < self->length; i++) {
      self->values[i] = fmin(self->values[i], operand);
   }
}

__attribute__((nonnull))
void tensor_float_min(tensor_float_t* self, tensor_float_t* operand) {
   assert(self);
   assert(operand);
   int len = min(self->length, operand->length);
   for (int i = 0; i < len; i++) {
      self->values[i] = fmin(self->values[i], operand->values[i]);
   }
}

//__attribute__((optimize("unroll-loops")))
//__attribute__((optimize("fast-math")))
float tensor_float_hsum1(tensor_float_t* self) {
   assert(self);
   float accum = 0.0;
   for (int i = 0; i < self->length; i++) {
      accum += self->values[i];
   }
   return accum;
}

// takes a tensor of weights and sums those in to a smaller operand; so
// if 'self' is 10 weights and 5 for each neuron, the operand will have
// a length of two, and be filled with the sums of those weights. this
// lets us calculate the amount of "fault" a given neuron is at.
void tensor_float_despread_sum(tensor_float_t* self, tensor_float_t* operand)
{
   assert(self);
   assert(operand);
   assert((self->length % operand->length) == 0);

   int stripe = self->length / operand->length;
   int at = 0;
   for (int i = 0; i < operand->length; i++) {
      for (int j = 0; j < stripe; j++) {
	 operand->values[i] += self->values[at++];
      }
   }
}

void tensor_float_free(tensor_float_t* self) {
   if (!self) return;
   if (self->values) free(self->values);
   free(self);
}

