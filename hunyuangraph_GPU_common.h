#ifndef _H_GPU_COMMON
#define _H_GPU_COMMON

// #include "hunyuangraph_struct.h"
// #include "hunyuangraph_common.h"
// #include "hunyuangraph_admin.h"

// #include <cuda_runtime.h>
// #include <cstdint> 
#include <curand_kernel.h>

__global__ void initializeCurand(unsigned long long seed, unsigned long long offset, unsigned long long nvtxs, curandStateXORWOW *devStates) 
{
    int ii = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, offset, nvtxs, &devStates[ii]);
}

__device__ int get_random_number_range(int range, curandState *localState) 
{
    float randNum = curand_uniform(localState);
    return (int)(randNum * range);
}

#endif