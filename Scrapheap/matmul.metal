#include <metal_stdlib>
using namespace metal;

kernel void matmul_baseline(device const float* A, // M x K
                            device const float* B, // K x N
                            device float* C,
                            constant uint& M,
                            constant uint& N,
                            constant uint& K,
                            uint2 pos [[thread_position_in_grid]])
{
    float sum = 0;
    for (uint k = 0; k < K; ++k) {
        sum += A[pos.y * K + k]
             * B[k * N + pos.x];
    }
    C[pos.y * N + pos.x] = sum;
}

kernel void matmul_simdgroup(device const float* A, // M x K
                             device const float* B, // K x N
                             device float* C,
                             constant uint& M,
                             constant uint& N,
                             constant uint& K,
                             uint2 pos [[thread_position_in_grid]])
{
    ulong2 originC = {(pos.x/8) * 8, (pos.y/8) * 8};
    simdgroup_float8x8 tileA, tileB;
    simdgroup_float8x8 tileC = make_filled_simdgroup_matrix<float, 8>(0);
    for (uint k = 0; k < K; k += 8) {
        ulong2 originA = {k, originC.y};
        ulong2 originB = {originC.x, k};
        simdgroup_load(tileA, A, K, originA);
        simdgroup_load(tileB, B, N, originB);
        simdgroup_multiply_accumulate(tileC, tileA, tileB, tileC);
    }
    simdgroup_store(tileC, C, N, originC);
}

#define TG_K 16

kernel void matmul_threadgroup(device const float* A, // M x K
                               device const float* B, // K x N
                               device float* C,
                               constant uint& M,
                               constant uint& N,
                               constant uint& K,
                               uint2 pos [[thread_position_in_grid]],
                               uint2 tgPos [[thread_position_in_threadgroup]])
{
    threadgroup float tgA[TG_K * TG_K], tgB[TG_K * TG_K];

    float sum = 0;
    for (uint k = 0; k < K; k += TG_K) {
        tgA[tgPos.y * TG_K + tgPos.x] = A[pos.y * K + (k + tgPos.x)];
        tgB[tgPos.y * TG_K + tgPos.x] = B[(k + tgPos.y) * N + pos.x];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < TG_K; ++i) {
            sum += tgA[tgPos.y * TG_K + i] * tgB[i * TG_K + tgPos.x];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    C[pos.y * N + pos.x] = sum;
}

// Version where threadgroup memory is transposed -- it'll be threadgroup load limited.
kernel void matmul_threadgroup_t(device const float* A, // M x K
                                 device const float* B, // K x N
                                 device float* C,
                                 constant uint& M,
                                 constant uint& N,
                                 constant uint& K,
                                 uint2 pos [[thread_position_in_grid]],
                                 uint2 tgPos [[thread_position_in_threadgroup]])
{
    threadgroup float tgA[TG_K * TG_K], tgB[TG_K * TG_K];

    float sum = 0;
    for (uint k = 0; k < K; k += TG_K) {
        tgA[tgPos.y * TG_K + tgPos.x] = A[pos.y * K + (k + tgPos.x)];
        tgB[tgPos.x * TG_K + tgPos.y] = B[(k + tgPos.y) * N + pos.x];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < TG_K; ++i) {
            sum += tgA[tgPos.y * TG_K + i] * tgB[tgPos.x * TG_K + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    C[pos.y * N + pos.x] = sum;
}

#define T1D_TG 8
#define T1D_TM 4

kernel void matmul_1d_threadtile(device const float* A, // M x K
                                 device const float* B, // K x N
                                 device float* C,
                                 constant uint& M,
                                 constant uint& N,
                                 constant uint& K,
                                 uint2 pos [[thread_position_in_grid]],
                                 uint2 tgPos [[thread_position_in_threadgroup]])
{
    threadgroup float tgA[T1D_TG * T1D_TG * T1D_TM], tgB[T1D_TG * T1D_TG];

    float sum[T1D_TM] = {0.};
    for (uint k = 0; k < K; k += T1D_TG) {
        for (uint t = 0; t < T1D_TM; ++t) {
            tgA[(tgPos.y*T1D_TM+t) * T1D_TG + tgPos.x] = A[(pos.y*T1D_TM+t) * K + (k + tgPos.x)];
        }
        tgB[tgPos.y * T1D_TG + tgPos.x] = B[(k + tgPos.y) * N + pos.x];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < T1D_TG; ++i) {
            for (uint t = 0; t < T1D_TM; ++t) {
                sum[t] += tgA[(tgPos.y*T1D_TM+t) * T1D_TG + i] * tgB[i * T1D_TG + tgPos.x];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint t = 0; t < T1D_TM; ++t) {
        C[(pos.y*T1D_TM+t) * N + pos.x] = sum[t];
    }
}

#define T2D_TG 8
#define T2D_TM 8
#define T2D_TN 8

kernel void matmul_2d_threadtile(device const float* A, // M x K
                                 device const float* B, // K x N
                                 device float* C,
                                 constant uint& M,
                                 constant uint& N,
                                 constant uint& K,
                                 uint2 pos [[thread_position_in_grid]],
                                 uint2 tgPos [[thread_position_in_threadgroup]])
{
    threadgroup float tgA[T2D_TG * T2D_TG * T2D_TM], tgB[T2D_TG * T2D_TG * T2D_TN];

    float sum[T2D_TM][T2D_TN] = {{0.}};
    for (uint k = 0; k < K; k += T2D_TG) {
        for (uint t = 0; t < T2D_TM; ++t) {
            tgA[(tgPos.y*T2D_TM+t) * T2D_TG + tgPos.x] = A[(pos.y*T2D_TM+t) * K + (k + tgPos.x)];
        }
        for (uint t = 0; t < T2D_TN; ++t) {
            tgB[tgPos.y*T2D_TG*T2D_TN + tgPos.x*T2D_TN + t] = B[(k + tgPos.y) * N + pos.x*T2D_TN + t];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < T2D_TG; ++i) {
            for (uint t = 0; t < T2D_TM; ++t) {
                for (uint v = 0; v < T2D_TM; ++v) {
                    sum[t][v] += tgA[(tgPos.y*T2D_TM+t) * T2D_TG + i] * tgB[i*T2D_TG*T2D_TN + tgPos.x*T2D_TN + v];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint t = 0; t < T2D_TM; ++t) {
        for (uint v = 0; v < T2D_TN; ++v) {
            C[(pos.y*T2D_TM+t) * N + pos.x*T2D_TN + v] = sum[t][v];
        }
    }
}
