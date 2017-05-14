#pragma once

typedef struct _tensor_float_t {
   float* values;
   int length;
} tensor_float_t;

// NB this works for now, although if we start doing weird things with
// memory it's going to cause some problems
#define tensor_float_set_at(self, index, value) \
	 self->values[index] = value
#define tensor_float_get_at(self, index, value) \
	 self->values[index]

int tensor_float_copy(tensor_float_t* dest, tensor_float_t* source, int dest_offset, int source_offset, int length);
tensor_float_t* tensor_float_flat_new(int length);
void tensor_float_spread(tensor_float_t* source, tensor_float_t* weights, tensor_float_t* destination);
void tensor_float_set1(tensor_float_t* self, float operand);
void tensor_float_set(tensor_float_t* self, tensor_float_t* operand);
void tensor_float_mul1(tensor_float_t* self, float operand);
void tensor_float_mul(tensor_float_t* self, tensor_float_t* operand);
void tensor_float_div1(tensor_float_t* self, float operand);
void tensor_float_div(tensor_float_t* self, tensor_float_t* operand);
void tensor_float_add1(tensor_float_t* self, float operand);
void tensor_float_add(tensor_float_t* self, tensor_float_t* operand);
void tensor_float_sub1(tensor_float_t* self, float operand);
void tensor_float_sub(tensor_float_t* self, tensor_float_t* operand);
void tensor_float_max1(tensor_float_t* self, float operand);
void tensor_float_max(tensor_float_t* self, tensor_float_t* operand);
void tensor_float_min1(tensor_float_t* self, float operand);
void tensor_float_min(tensor_float_t* self, tensor_float_t* operand);
float tensor_float_hsum1(tensor_float_t* self);
void tensor_float_free(tensor_float_t* self);

