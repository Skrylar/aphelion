
# TODO replace seeding with a simple key scheduler; that way we aren't
# dependent on weird environmental factors for initializing random
# numbers.

import random

# https://en.wikipedia.org/wiki/Multiply-with-carry
# C99 Complementary Multiply With Carry generator

const
   # CMWC working parts
   CMWC_CYCLE = 4096      # as Marsaglia recommends
   CMWC_C_MAX = 809430660 # as Marsaglia recommends
   a = uint64(18782)      # as Marsaglia recommends
   m = uint32(0xfffffffe) # as Marsaglia recommends

type
   Cmwc* = ref object of Random
      Q: array[0..(CMWC_CYCLE-1), uint32]
      c: uint32 ## must be limited with CMWC_C_MAX
      i: cuint

# TODO replace this stuff with a key derivation function, so we aren't
# screwing around with system state every time we re-seed

proc rand(): cint {.importc.}
proc srand(seed: cuint) {.importc.}

proc rand32(): uint32 =
   result = uint32(rand())
   result = result shl 16 or uint32(rand())

method seed*(state: Cmwc, seed: uint) =
   ## Init the state with seed

   srand cuint(seed)

   for i in 0..(CMWC_CYCLE-1):
      state.Q[i] = rand32()

   state.c = rand32()
   while state.c >= uint32(CMWC_C_MAX):
      state.c = rand32()

   state.i = CMWC_CYCLE - 1;

method next_u32(state: Cmwc): uint32 =
   state.i = (state.i + 1) and (CMWC_CYCLE - 1);
   var t: uint64 = a * state.Q[state.i] + state.c;

   # Let c = t / 0xfffffff, x = t mod 0xffffffff

   state.c = uint32(t shr 32)
   var x: uint32 = uint32(t) + state.c;

   if x < state.c:
      inc x
      inc state.c

   state.Q[state.i] = m - x;
   return state.Q[state.i]

when isMainModule:
   var rng = Cmwc()
   rng.seed(32)
   echo rng.next_u32
   echo rng.next_u32
   echo rng.next_float
   echo rng.next_float

