
// TODO replace seeding with a simple key scheduler; that way we aren't
// dependent on weird environmental factors for initializing random
// numbers.

#include "random.h"

// https://en.wikipedia.org/wiki/Multiply-with-carry
// C99 Complementary Multiply With Carry generator
#include <stdio.h>
#include <stdlib.h>

// CMWC working parts
#define CMWC_CYCLE 4096 // as Marsaglia recommends
#define CMWC_C_MAX 809430660 // as Marsaglia recommends

struct cmwc {
   random_t super;

   uint32_t Q[CMWC_CYCLE];
   uint32_t c;	// must be limited with CMWC_C_MAX
   unsigned i;
};

static uint32_t rand32(void)
{
   uint32_t result = rand();
   return result << 16 | rand();
}

// Init the state with seed
static void cmwc_init(struct cmwc *state, unsigned int seed)
{
   srand(seed);        

   for (int i = 0; i < CMWC_CYCLE; i++) {
      state->Q[i] = rand32();
   }

   do {
      state->c = rand32();
   } while (state->c >= CMWC_C_MAX);

   state->i = CMWC_CYCLE - 1;
}

// CMWC engine
static uint32_t randCMWC(struct cmwc *state)  //EDITED parameter *state was missing
{
   uint64_t const a = 18782; // as Marsaglia recommends
   uint32_t const m = 0xfffffffe; // as Marsaglia recommends
   uint64_t t;
   uint32_t x;

   state->i = (state->i + 1) & (CMWC_CYCLE - 1);
   t = a * state->Q[state->i] + state->c;
   /* Let c = t / 0xfffffff, x = t mod 0xffffffff */
   state->c = t >> 32;
   x = t + state->c;
   if (x < state->c) {
      x++;
      state->c++;
   }
   return state->Q[state->i] = m - x;
}

static float cmwc_next_float(struct cmwc* self) {
   return randCMWC(self) / (float)0xFFFFFFFF;
}

static void cmwc_free(struct cmwc* self) {
   free(self);
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wincompatible-pointer-types"
static random_class_t random_class = {
   &cmwc_free,
   &cmwc_init,
   &randCMWC,
   &cmwc_next_float,
};
#pragma GCC diagnostic pop

random_t* random_cmwc_new(unsigned int seed) {
   struct cmwc* self = malloc(sizeof(struct cmwc));
   if (!self) goto no_self;
   self->super.cls = &random_class;
   cmwc_init(self, seed);

   return (random_t*)self;
no_self:
   return 0;
}

