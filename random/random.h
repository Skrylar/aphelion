#pragma once

#include <stdint.h>
#include <assert.h>

typedef uint32_t (*rand32_f)(void* self);
typedef float (*randf_f)(void* self);
typedef void (*rand_free_f)(void* self);
typedef void (*rand_init_f)(void* self, unsigned int seed);

typedef struct _random_class_t {
   rand_free_f free;
   rand_init_f init;
   rand32_f rand32;
   randf_f randf;
} random_class_t;

typedef struct _random_t {
   random_class_t* cls;
} random_t;

inline uint32_t random_next_u32(random_t* self) {
   assert(self);
   return self->cls->rand32(self);
}

inline float random_next_float(random_t* self) {
   assert(self);
   return self->cls->randf(self);
}

inline void random_free(random_t* self) {
   assert(self);
   self->cls->free(self);
}

