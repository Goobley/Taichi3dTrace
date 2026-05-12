import taichi as ti
from taichi.math import ivec2, ivec3

# https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
# // "Insert" a 0 bit after each of the 16 low bits of x
@ti.func
def part_1_by_1(x: ti.u32) -> ti.u32:
    x &= ti.u32(0x0000FFFF)                  # x = ---- ---- ---- ---- fedc ba98 7654 3210
    x = (x ^ (x << 8)) & ti.u32(0x00FF00FF)  # x = ---- ---- fedc ba98 ---- ---- 7654 3210
    x = (x ^ (x << 4)) & ti.u32(0x0F0F0F0F)  # x = ---- fedc ---- ba98 ---- 7654 ---- 3210
    x = (x ^ (x << 2)) & ti.u32(0x33333333)  # x = --fe --dc --ba --98 --76 --54 --32 --10
    x = (x ^ (x << 1)) & ti.u32(0x55555555)  # x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
    return x

# "Insert" two 0 bits after each of the 10 low bits of x
@ti.func
def part_1_by_2(x: ti.u32) -> ti.u32:
  x &= ti.u32(0x000003ff)                  # x = ---- ---- ---- ---- ---- --98 7654 3210
  x = (x ^ (x << 16)) & ti.u32(0xff0000ff) # x = ---- --98 ---- ---- ---- ---- 7654 3210
  x = (x ^ (x <<  8)) & ti.u32(0x0300f00f) # x = ---- --98 ---- ---- 7654 ---- ---- 3210
  x = (x ^ (x <<  4)) & ti.u32(0x030c30c3) # x = ---- --98 ---- 76-- --54 ---- 32-- --10
  x = (x ^ (x <<  2)) & ti.u32(0x09249249) # x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
  return x

# Inverse of Part1By1 - "delete" all odd-indexed bits
@ti.func
def compact_1_by_1(x: ti.u32) -> ti.u32:
    x &= ti.u32(0x55555555)                  # x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
    x = (x ^ (x >> 1)) & ti.u32(0x33333333)  # x = --fe --dc --ba --98 --76 --54 --32 --10
    x = (x ^ (x >> 2)) & ti.u32(0x0F0F0F0F)  # x = ---- fedc ---- ba98 ---- 7654 ---- 3210
    x = (x ^ (x >> 4)) & ti.u32(0x00FF00FF)  # x = ---- ---- fedc ba98 ---- ---- 7654 3210
    x = (x ^ (x >> 8)) & ti.u32(0x0000FFFF)  # x = ---- ---- ---- ---- fedc ba98 7654 3210
    return x

@ti.func
def compact_1_by_2(x: ti.u32) -> ti.u32:
  x &= ti.u32(0x09249249)                  # x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
  x = (x ^ (x >>  2)) & ti.u32(0x030c30c3) # x = ---- --98 ---- 76-- --54 ---- 32-- --10
  x = (x ^ (x >>  4)) & ti.u32(0x0300f00f) # x = ---- --98 ---- ---- 7654 ---- ---- 3210
  x = (x ^ (x >>  8)) & ti.u32(0xff0000ff) # x = ---- --98 ---- ---- ---- ---- 7654 3210
  x = (x ^ (x >> 16)) & ti.u32(0x000003ff) # x = ---- ---- ---- ---- ---- --98 7654 3210
  return x


@ti.func
def encode_morton_2(x: ti.u32, z: ti.u32) -> ti.u32:
    """Encodes the lower 16 bits of two int32 into a uint32"""
    return (part_1_by_1(ti.u32(z)) << 1) + part_1_by_1(ti.u32(x))

@ti.func
def encode_morton_3(x: ti.u32, y: ti.u32, z: ti.u32) -> ti.u32:
    """Encodes the lower 10 bits of three int32 into a uint32"""
    return (
        (part_1_by_2(ti.u32(z)) << 2)
        + (part_1_by_2(ti.u32(y)) << 1)
        + part_1_by_2(ti.u32(x))
    )

@ti.func
def decode_morton_2(code: ti.u32) -> ivec2:
    """Decodes a uint32 morton code
    Returns: x, z as ivec2
    """
    return ivec2(compact_1_by_1(code >> 0), compact_1_by_1(code >> 1))  # x, z

@ti.func
def decode_morton_3(code: ti.u32) -> ivec3:
    """Decodes a uint32 3d morton code
    Returns: x, y, z as ivec3
    """
    return ivec3(compact_1_by_2(code >> 0), compact_1_by_2(code >> 1), compact_1_by_2(code >> 2))  # x, y, z