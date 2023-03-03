#include <metal_stdlib>
using namespace metal;

kernel void sum_float(device float* in,
                      device float* out,
                      uint tid [[ threadgroup_position_in_grid ]]) {
    out[tid] = simd_sum(in[tid]);
}

kernel void sum_int(device int* in,
                    device int* out,
                    uint tid [[ threadgroup_position_in_grid ]]) {
    out[tid] = simd_sum(in[tid]);
}

kernel void sum_uint(device uint* in,
                    device uint* out,
                    uint tid [[ threadgroup_position_in_grid ]]) {
    out[tid] = simd_sum(in[tid]);
}

kernel void prefix_sum_float(device float* in,
                             device float* out,
                             uint tid [[ threadgroup_position_in_grid ]]) {
    out[tid] = simd_prefix_exclusive_sum(in[tid]);
}
