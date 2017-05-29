
type
   Random* = ref object {.inheritable.}

method next_u32*(self: Random): uint32 {.base.} =
   return 0

method next_float*(self: Random): float {.base.} =
   return float(next_u32(self)) / float(0xFFFFFFFF)

method seed*(self: Random, new_seed: uint) {.base.} =
   discard

