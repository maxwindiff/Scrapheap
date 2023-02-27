//
//  ParallelReduce.metal
//
//  Created by Matthew Kieber-Emmons on 03/11/2021.
//  Copyright © 2021 Matthew Kieber-Emmons. All rights reserved.
//  This work is for educational purposes only and cannot be used without consent.
//

#include <metal_stdlib>
using namespace metal;

////////////////////////////////////////////////////////////////
//  MARK: - FUNCTION CONSTANTS
////////////////////////////////////////////////////////////////
// these constants control the code paths at pipeline creation
constant int LOCAL_ALGORITHM [[function_constant(0)]];
constant int GLOBAL_ALGORITHM [[function_constant(1)]];
constant bool DISABLE_BOUNDS_CHECK [[function_constant(2)]];
constant bool USE_SHUFFLE [[function_constant(3)]];
////////////////////////////////////////////////////////////////

struct Sum {
    template <typename T, typename U> inline T operator()(thread const T& a, thread const U& b) const{return a + b;}
    template <typename T, typename U> inline T operator()(threadgroup const T& a, threadgroup const U& b) const{return a + b;}
    constexpr uint identity(){ return 0; }
};

////////////////////////////////////////////////////////////////
//  MARK: - MODIFIED LOAD
////////////////////////////////////////////////////////////////

//  this is a blocked read into registers without bounds checking
template<ushort GRAIN_SIZE, typename OPERATION, typename T, typename U>
static void LoadLocalReduceFromGlobal(thread U &value,
                                      const device T* input_data,
                                      const ushort lid) {
    OPERATION Op;
    value = input_data[lid * GRAIN_SIZE];
    for (ushort i = 1; i < GRAIN_SIZE; i++){
        value = Op(value,input_data[lid * GRAIN_SIZE + i]);
    }

}

//  this is a blocked read into registers with bounds checking
template<ushort GRAIN_SIZE, typename OPERATION, typename T, typename U>
static void LoadLocalReduceFromGlobal(thread U &value,
                                      const device T* input_data,
                                      const ushort lid,
                                      const uint n,
                                      const U substitution_value) {
    OPERATION Op;
    value = (lid * GRAIN_SIZE < n) ? input_data[lid * GRAIN_SIZE] : substitution_value;
    for (ushort i = 1; i < GRAIN_SIZE; i++){
        value = Op(value, (lid * GRAIN_SIZE + i < n) ?
                    input_data[lid * GRAIN_SIZE + i] : substitution_value);
    }
}

////////////////////////////////////////////////////////////////
//  MARK: - SIMDGROUP REDUCE
////////////////////////////////////////////////////////////////

template <typename OPERATION, typename T> static inline T
SimdgroupReduceSharedMem(volatile threadgroup T* shared, ushort lid){
    OPERATION Op;
    shared[lid] = Op(shared[lid],shared[lid + 16]);
    shared[lid] = Op(shared[lid],shared[lid +  8]);
    shared[lid] = Op(shared[lid],shared[lid +  4]);
    shared[lid] = Op(shared[lid],shared[lid +  2]);
    shared[lid] = Op(shared[lid],shared[lid +  1]);
    return shared[0];
}

template <typename OPERATION, typename T> static inline T
SimdgroupReduceShuffle(T value){
    OPERATION Op;
    value = Op(value,simd_shuffle_down(value, 16));
    value = Op(value,simd_shuffle_down(value,  8));
    value = Op(value,simd_shuffle_down(value,  4));
    value = Op(value,simd_shuffle_down(value,  2));
    value = Op(value,simd_shuffle_down(value,  1));
    return value;
}

////////////////////////////////////////////////////////////////
//  MARK: - THREADGROUP REDUCE
////////////////////////////////////////////////////////////////
// This kernel is a work efficent but moderately cost inefficient reduction in shared memory.
// Kernel is inspired by "Optimizing Parallel Reduction in CUDA" by Mark Harris:
// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
template <ushort BLOCK_SIZE, typename OPERATION, typename T, bool BROADCAST> static T
ThreadgroupReduceSharedMemAlgorithm(T value, threadgroup T* shared, const ushort lid){

    // copy values to shared memory
    shared[lid] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // initial reductions in shared memory
    OPERATION Op;
    if (BLOCK_SIZE >= 1024) {if (lid < 512) {shared[lid] = Op(shared[lid], shared[lid + 512]);} threadgroup_barrier(mem_flags::mem_threadgroup);}
    if (BLOCK_SIZE >=  512) {if (lid < 256) {shared[lid] = Op(shared[lid], shared[lid + 256]);} threadgroup_barrier(mem_flags::mem_threadgroup);}
    if (BLOCK_SIZE >=  256) {if (lid < 128) {shared[lid] = Op(shared[lid], shared[lid + 128]);} threadgroup_barrier(mem_flags::mem_threadgroup);}
    if (BLOCK_SIZE >=  128) {if (lid <  64) {shared[lid] = Op(shared[lid], shared[lid +  64]);} threadgroup_barrier(mem_flags::mem_threadgroup);}

    //  final reduction in shared warp
    if (lid < 32){
        
        //  we fold one more time
        if (BLOCK_SIZE >= 64) {
            shared[lid] = Op(shared[lid],shared[lid + 32]);
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        
        if (USE_SHUFFLE){
            value = SimdgroupReduceShuffle<OPERATION>(shared[lid]);
        }else{
            value = SimdgroupReduceSharedMem<OPERATION>(shared, lid);
        }
    }
    
    //  only result in thread0 is defined unless requested
    if (BROADCAST){
        if (USE_SHUFFLE){
            if (lid < 32){
                shared[lid] = value;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lid % 32 == 0){
                value = shared[lid / 32];
                value = simd_broadcast_first(value);
            }
        } else {
            //  raking write of results to shared memory
            if (lid < 32){
                for (short i = 0; i < BLOCK_SIZE / 32; i++){
                    shared[lid + i * 32] = value;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            value = shared[lid];
        }
    }
    
    return value;
    
}

// This kernel is a work and cost efficent rake in shared memory.
// Kernel is inspired by CUB library by NVIDIA
template <ushort BLOCK_SIZE, typename OPERATION, typename T, bool BROADCAST> static T
ThreadgroupReduceRakingAlgorithm(T value, threadgroup T* shared, const ushort lid){
    
    // copy values to shared memory
    shared[lid] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    //  first warp reduces all values
    if (lid < 32){
        
        OPERATION Op;

        //  interleaved addressing to reduce values into 0...31
        for (ushort i = 1; i < BLOCK_SIZE / 32; i++){
            shared[lid] = Op(shared[lid], shared[lid + 32 * i]);
        }

        //  final reduction
        if (USE_SHUFFLE){
            value = SimdgroupReduceShuffle<OPERATION>(shared[lid]);
        }else{
            value = SimdgroupReduceSharedMem<OPERATION>(shared, lid);
        }
    }
    
    //  only result in thread0 is defined unless requested
    if (BROADCAST){
        if (USE_SHUFFLE){
            if (lid < 32){
                shared[lid] = value;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lid % 32 == 0){
                value = shared[lid / 32];
                value = simd_broadcast_first(value);
            }
        } else {
            //  raking write of results to shared memory
            if (lid < 32){
                for (short i = 0; i < BLOCK_SIZE / 32; i++){
                    shared[lid + i * 32] = value;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            value = shared[lid];
        }
    }

    return value;

}

//  This is a highly parallel but not cost efficient algorithm,
//  which sometimes yields results back faster
template <ushort BLOCK_SIZE, typename OPERATION, typename T, bool BROADCAST> static T
ThreadgroupReduceCooperativeAlgorithm(T value, threadgroup T* shared, const ushort lid){
     
    OPERATION Op;

    //  first level of reduction in simdgroup
    if (USE_SHUFFLE){
        value = SimdgroupReduceShuffle<OPERATION>(value);
    }else{
        // copy values to shared memory
        shared[lid] = value;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        value = SimdgroupReduceSharedMem<OPERATION>(&shared[lid / 32 * 32], lid % 32);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    //  return early if our block size is 32
    if (BLOCK_SIZE == 32){
        if (BROADCAST) value = simd_broadcast_first(value);
        return value;
    }
    
    //  first simd lane writes to shared
    if (lid % 32 == 0)
        shared[lid / 32] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    //  final reduction in first simdgroup
    if (lid < 32){
        
        //  mask the values on copy
        value = (lid < BLOCK_SIZE / 32) ? shared[lid] : Op.identity();
        
        //  final reduction
        if (USE_SHUFFLE){
            value = SimdgroupReduceShuffle<OPERATION>(value);
        }else{
            shared[lid] = value;
            value = SimdgroupReduceSharedMem<OPERATION>(shared, lid);
        }
    }

    //  only result in thread0 is defined unless requested
    if (BROADCAST){
        if (USE_SHUFFLE){
            if (lid < 32){
                shared[lid] = value;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lid % 32 == 0){
                value = shared[lid / 32];
                value = simd_broadcast_first(value);
            }
        } else {
            //  raking write of results to shared memory
            if (lid < 32){
                for (short i = 0; i < BLOCK_SIZE / 32; i++){
                    shared[lid + i * 32] = value;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            value = shared[lid];
        }
    }
    
    return value;
}

template <ushort BLOCK_SIZE, typename OPERATION, typename T, bool BROADCAST = false> static T
ThreadgroupReduce(T value, threadgroup T* shared, const ushort lid) {
    
    // reduce by threadgroup
    T sum = 0;
    switch (LOCAL_ALGORITHM){
        case 0:
            sum = ThreadgroupReduceSharedMemAlgorithm<BLOCK_SIZE,OPERATION,T,BROADCAST>(value, shared, lid);
            break;
        case 1:
            sum = ThreadgroupReduceRakingAlgorithm<BLOCK_SIZE,OPERATION,T,BROADCAST>(value, shared, lid);
            break;
        case 2:
            sum = ThreadgroupReduceCooperativeAlgorithm<BLOCK_SIZE,OPERATION,T,BROADCAST>(value, shared, lid);
            break;
    }
    return sum;
}

////////////////////////////////////////////////////////////////
//  MARK: - REDUCTION KERNELS
////////////////////////////////////////////////////////////////

template<ushort BLOCK_SIZE, ushort GRAIN_SIZE, typename OPERATION = Sum, typename T, typename U, typename V> kernel void
ReduceKernel(device V* output_data,
             device const T* input_data,
             constant uint& n,
             threadgroup U* scratch,
             uint group_id [[ threadgroup_position_in_grid ]],
             ushort local_id [[ thread_index_in_threadgroup ]]) {

    uint base_id = group_id * BLOCK_SIZE * GRAIN_SIZE;

    // reduce during the load from global
    U value;
    if (DISABLE_BOUNDS_CHECK) {
        LoadLocalReduceFromGlobal<GRAIN_SIZE,OPERATION>(value, &input_data[base_id], local_id);
    } else {
        OPERATION Op;
        LoadLocalReduceFromGlobal<GRAIN_SIZE,OPERATION>(value, &input_data[base_id], local_id, n - base_id, Op.identity());
    }

    // reduce the values from each thread in the threadgroup
    value = ThreadgroupReduce<BLOCK_SIZE,OPERATION,U>(value, scratch, local_id);

    // write result for this threadgroup to global memory
    if (local_id == 0)
        output_data[group_id] = value;
}

template [[host_name("reduce_sum_uint32_256threads_4way")]]
kernel void ReduceKernel<256,4,Sum,uint,uint,uint>(device uint*, device const uint*, constant uint&, threadgroup uint*, uint, ushort);
